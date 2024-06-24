"""
wild mixture of
https://github.com/autonomousvision/convolutional_occupancy_networks
for implementing occupancy function
"""

import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from torch import distributions as dist
from torchvision import utils as vtils
from torch.optim import AdamW
from accelerate import Accelerator

from losses.lpips import LPIPS
from utils.general_utils import exists, linear_kl_coeff, convert_to_coord_format_3d
from utils.sr_utils import SpectralNormCalculator
from evals.eval import test_rfvd, test_psnr


class D2CTrainer(object):
    def __init__(
            self,
            args,
            vaemodel,
            mlp,
            data,
            test_data=None,
            criterion=None,
            ):
        super().__init__()

        ## Accelerator
        self.accelerator = Accelerator(
                split_batches = False,
                mixed_precision = 'fp16' if args.use_fp16 else 'no'
                )
        self.accelerator.native_amp = args.amp

        self.data = data
        self.test_data = test_data
        self.args = args
        self.criterion = criterion

        ## models
        self.vaemodel = vaemodel
        self.mlp = mlp

        ## loss config
        self.train_lr = args.lr
        self.warmup_epochs = args.loss_config.warmup_epochs
        self.epochs = args.loss_config.epochs
        self.save_and_sample_every = args.loss_config.save_and_sample_every
        self.warmup_iters = len(data) * self.warmup_epochs
        self.num_total_iters = len(data) * self.epochs
        self.kl_anneal = args.loss_config.kl_anneal
        self.kl_anneal_portion = args.loss_config.kl_anneal_portion
        self.kl_const_portion = args.loss_config.kl_const_portion
        self.kl_const_coeff = args.loss_config.kl_const_coeff
        self.kl_max_coeff = args.loss_config.kl_max_coeff
        self.sn_reg_weight_decay = args.loss_config.sn_reg_weight_decay
        self.sn_reg_weight_decay_anneal = args.loss_config.sn_reg_weight_decay_anneal
        self.sn_reg_weight_decay_init = args.loss_config.sn_reg_weight_decay_init
        self.sn_reg = args.loss_config.sn_reg
        self.perceptual_loss = LPIPS().eval()
        self.gradient_accumulate_every = args.loss_config.gradient_accumulate_every
        if args.loss_config.sn_reg:
            print('Spectral Normalization Regularization Activated!')
            self.sn_caculator = SpectralNormCalculator()
            self.sn_caculator.add_conv_layers(vaemodel)
            self.sn_caculator.add_bn_layers(vaemodel)
        else:
            self.sn_caculator = None
            print('No SN Reg!')

        ## Optimizer
        self.opt = AdamW(list(vaemodel.parameters()) + list(mlp.parameters()), lr = args.lr, betas = (0.9, 0.99))
        if args.loss_config.lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt, (self.epochs - self.warmup_epochs - 1), eta_min = 1e-6
            )
        else:
            self.scheduler = None

        ## Discriminator
        if args.loss_config.adversarial:
            assert self.criterion is not None
            self.opt_d = AdamW(list(self.criterion.discriminator_2d.parameters()) + list(self.criterion.discriminator_3d.parameters()), 
                             lr=args.lr, 
                             betas=(0.5, 0.9))
        else:
            self.criterion = None
            self.opt_d = None

        ## Initialize
        self.step = 0
        self.current_iters = 0
    
        ## Load from previous checkpoint
        if args.resume:
            print('Load checkpoint from previous training!')
            self.load(os.path.join(args.data_config.save_pth, 'model-last.pt'))
            print('Current Epochs : ', self.step)
            print('Current iters : ', self.current_iters)

        self.data, self.vaemodel, self.mlp, self.opt, self.perceptual_loss = self.accelerator.prepare(self.data, self.vaemodel, self.mlp, self.opt, self.perceptual_loss)
        if args.loss_config.adversarial:
            self.criterion, self.opt_d = self.accelerator.prepare(self.criterion, self.opt_d)

        if exists(self.sn_caculator):
            self.sn_caculator = self.accelerator.prepare(self.sn_caculator)

        ## Save directory
        self.results_folder = args.data_config.save_pth
        os.makedirs(self.results_folder, exist_ok=True)
        self.results_pth = os.path.join(self.results_folder, 'results')
        os.makedirs(self.results_pth, exist_ok=True)
        
    def save(self, step = 0):
        if not self.accelerator.is_main_process:
            return
        data = {
                'step' : self.step,
                'current_iters' : self.current_iters,
                'vaemodel' : self.accelerator.get_state_dict(self.vaemodel),
                'mlp' : self.accelerator.get_state_dict(self.mlp),
                'opt' : self.opt.state_dict(),
                'scaler' : self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
                'opt_sched' : self.scheduler.state_dict() if exists(self.scheduler) else None,
                'sn' : self.sn_caculator.state_dict() if exists(self.sn_caculator) else None,
                'opt_d' : self.opt_d.state_dict() if exists(self.opt_d) else None,
                'criterion' : self.accelerator.get_state_dict(self.criterion) if exists(self.criterion) else None,
                }
        torch.save(data, os.path.join(self.results_folder, 'model-{}.pt'.format(step)))
        torch.save(data, os.path.join(self.results_folder, 'model-last.pt'.format(step)))
 
    def load(self, pth):
        data = torch.load(pth, map_location = 'cpu')
        self.vaemodel.load_state_dict(data['vaemodel'])
        self.mlp.load_state_dict(data['mlp'])
        self.step = data['step']
        self.current_iters = data['current_iters']
        self.opt.load_state_dict(data['opt'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

        if exists(self.scheduler) and exists(data['opt_sched']):
            self.scheduler.load_state_dict(data['opt_sched'])
        
        if exists(self.sn_caculator) and data['sn'] is not None:
            self.sn_caculator.load_state_dict(data['sn'], device=self.accelerator.device)

        try:
            self.opt_d.load_state_dict(data['opt_d'])
            self.criterion.load_state_dict(data['criterion'])
        except:
            print('Not found criterion!')
            pass

    def train(self):
        device = self.accelerator.device
        optimizer_idx = True
        coords = convert_to_coord_format_3d(1, 256, 256, 16, device = device, hstart=-255/256, hend=255/256, 
                                            wstart=-255/256, wend=255/256, tstart = -15/16, tend = 15/16)
        
        with tqdm(initial = self.step, total = self.epochs) as pbar:
            while self.step < self.epochs:
                if self.step < self.warmup_epochs:
                    lr = self.train_lr * float(self.step + 1) / self.warmup_epochs
                    for param_group in self.opt.param_groups:
                        param_group['lr'] = lr
                
                if self.step > self.warmup_epochs and self.scheduler is not None:
                    self.scheduler.step()

                self.vaemodel.train()
                self.mlp.train()

                for idx, (x,_) in enumerate(self.data):
                    assert len(x.shape) == 5
                    x = (x / 127.5) - 1
                    x = x.permute(0, 2, 1, 3, 4).contiguous()
                    batch, channel, t, h, w = x.shape
                    
                    #self.opt.zero_grad()
                    with self.accelerator.autocast():
                        if isinstance(self.vaemodel, torch.nn.parallel.DistributedDataParallel):
                            posterior_xy, posterior_yt, posterior_xt = self.vaemodel.module.encode(x)
                            xy, yt, xt = posterior_xy.sample(), posterior_yt.sample(), posterior_xt.sample()
                            b, c = xy.shape[0], xy.shape[1]
                            z = torch.cat([xy.reshape(b, c, -1), xt.reshape(b, c, -1), yt.reshape(b, c, -1)], dim = 2)
                            pe_xy, pe_yt, pe_xt = self.vaemodel.module.decode(z)
                        else:
                            posterior_xy, posterior_yt, posterior_xt = self.vaemodel.encode(x)
                            b, c = xy.shape[0], xy.shape[1]
                            z = torch.cat([xy.reshape(b, c, -1), xt.reshape(b, c, -1), yt.reshape(b, c, -1)], dim = 2)
                            pe_xy, pe_yt, pe_xt = self.vaemodel.decode(z)
                        output = self.mlp(coords, (pe_xy, pe_yt, pe_xt))
                        if optimizer_idx:
                            ## Recon Loss
                            recon_loss = torch.sum(torch.abs(output.contiguous() - x.contiguous()), dim = (1,2,3,4))
                            recon_loss = torch.mean(recon_loss)

                            ## Perceptual loss
                            p_coeff = 1.
                            frame_idx = torch.randint(0, t, [batch]).to(device)
                            frame_idx_selected = frame_idx.reshape(-1, 1, 1, 1, 1).repeat(1, channel, 1, h, w)
                            inputs_2d = torch.gather(x, 2, frame_idx_selected).squeeze(2)
                            recon_2d = torch.gather(output, 2, frame_idx_selected).squeeze(2)
                            p_loss = self.perceptual_loss(inputs_2d.contiguous(), recon_2d.contiguous()).mean()

                            ## KL loss
                            kld_xy = posterior_xy.kl()
                            kld_yt = posterior_yt.kl()
                            kld_xt = posterior_xt.kl()
                            kld = kld_xy + kld_yt + kld_xt
                            kld_loss = torch.mean(kld)
                            if self.kl_anneal:
                                kl_coeff = linear_kl_coeff(self.current_iters, self.kl_anneal_portion * self.num_total_iters,
                                                                self.kl_const_portion * self.num_total_iters, self.kl_const_coeff, self.kl_max_coeff)
                            else:
                                kl_coeff = self.kl_max_coeff

                            total_loss = recon_loss + p_coeff * p_loss + kl_coeff * kld_loss
            
                            # SN regularization
                            if self.sn_reg:
                                norm_loss = self.sn_caculator.spectral_norm_parallel()
                                bn_loss = self.sn_caculator.batchnorm_loss()
                                if self.sn_reg_weight_decay_anneal:
                                    wdn_coeff = (1. - kl_coeff) * np.log(self.sn_reg_weight_decay_init) + kl_coeff * np.log(self.sn_reg_weight_decay)
                                    wdn_coeff = np.exp(wdn_coeff)
                                else:
                                    wdn_coeff = self.sn_reg_weight_decay
                                
                                total_loss += norm_loss * wdn_coeff + bn_loss * wdn_coeff
                        else:
                            total_loss = 0.
                        
                        if self.args.loss_config.adversarial:
                            total_loss += self.criterion(x, output, optimizer_idx)
                        total_loss = total_loss / self.gradient_accumulate_every
                    pbar.set_description('recon : {:.4f} \ p_loss : {:.4f} \ kl : {:.4f}'.format(recon_loss, p_loss.item(), kld_loss.item()))
                    self.accelerator.backward(total_loss)
                    self.current_iters += 1
                    self.accelerator.wait_for_everyone()

                    if self.current_iters % self.gradient_accumulate_every == self.gradient_accumulate_every - 1:
                        if optimizer_idx:
                            self.opt.step()
                            self.opt.zero_grad()
                            if self.opt_d is not None:
                                self.opt_d.zero_grad()
                        else:
                            assert self.opt_d is not None
                            self.opt_d.step()
                            self.opt_d.zero_grad()
                            self.opt.zero_grad()
                        if self.args.loss_config.adversarial:
                            if optimizer_idx:
                                optimizer_idx = False
                            else:
                                optimizer_idx = True
                        self.accelerator.wait_for_everyone()
                
                if self.step % self.save_and_sample_every == 0 and self.accelerator.is_main_process:
                    ## R-FVD
                    if self.test_data is not None:
                        fvd = test_rfvd(self.vaemodel, self.mlp, coords, self.test_data, device, self.accelerator)
                        print('FVD:', fvd)
                    else:
                        pass
                
                self.accelerator.wait_for_everyone()
                
                if self.step % self.save_and_sample_every == 0 and self.accelerator.is_main_process:
                    f = output.shape[2]
                    step_save_pth = os.path.join(self.results_pth, 'step{}'.format(self.step))
                    gt_step_save_pth = os.path.join(self.results_pth, 'gt{}'.format(self.step))
                    os.makedirs(step_save_pth, exist_ok=True)
                    os.makedirs(gt_step_save_pth, exist_ok=True)
                    for ci in range(f):
                        vtils.save_image(output[:,:,ci], os.path.join(step_save_pth, 'sample-gen-{}-{}.png'.format(ci, self.step)), normalize = True, scale_each=True)

                    for ci in range(f):
                        vtils.save_image(x[:,:,ci], os.path.join(gt_step_save_pth, 'gt-{}-{}.png'.format(ci, self.step)), normalize = True, scale_each=True)
                    self.save(step = self.step)

                self.accelerator.wait_for_everyone()
                self.step += 1
                pbar.update(1)

    def eval(self):
        device = self.accelerator.device
        coords = convert_to_coord_format_3d(1, 256, 256, 16, device = device, hstart=-255/256, hend=255/256, 
                                            wstart=-255/256, wend=255/256, tstart = -15/16, tend = 15/16)
        if self.test_data is not None:
            print('Evaluating rFVD!')
            rfvd = test_rfvd(self.vaemodel, self.mlp, coords, self.test_data, device)
            print('rFVD:', rfvd)
        else:
            print('NO Test Dataset')
            pass

import os
import torch
import numpy as np
from tqdm import tqdm

from torchvision import utils as vtils
from torch.optim import AdamW
from accelerate import Accelerator

from losses.lpips import LPIPS
from utils.general_utils import exists, symmetrize_image_data, multiscale_image_transform, linear_kl_coeff, convert_to_coord_format_2d, cycle
from utils.sr_utils import SpectralNormCalculator
from evals.eval import test_rfid


class D2CTrainer(object):
    def __init__(
            self,
            args,
            model,
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
        self.model = model
        self.mlp = mlp

        ## loss config
        self.multiscale = args.loss_config.multiscale
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
        self.test_resolution = args.data_config.test_resolution
        if args.loss_config.sn_reg:
            print('Spectral Normalization Regularization Activated!')
            self.sn_caculator = SpectralNormCalculator()
            self.sn_caculator.add_conv_layers(model)
            self.sn_caculator.add_bn_layers(model)
        else:
            self.sn_caculator = None
            print('No SN Reg!')

        ## Optimizer
        self.opt = AdamW(list(model.parameters()) + list(mlp.parameters()), lr = args.lr, betas = (0.9, 0.99))
        if args.loss_config.lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt, (self.epochs - self.warmup_epochs - 1), eta_min = 1e-6
            )
        else:
            self.scheduler = None

        ## Discriminator
        if args.loss_config.adversarial:
            assert self.criterion is not None
            self.opt_d = AdamW(self.criterion.discriminator_2d.parameters(), 
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

        self.data, self.model, self.mlp, self.opt, self.perceptual_loss = self.accelerator.prepare(self.data, self.model, self.mlp, self.opt, self.perceptual_loss)
        #self.data = cycle(self.data)
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
                'model' : self.accelerator.get_state_dict(self.model),
                'mlp' : self.accelerator.get_state_dict(self.mlp),
                'opt' : self.opt.state_dict(),
                'scaler' : self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
                'opt_sched' : self.scheduler.state_dict() if exists(self.scheduler) else None,
                'sn' : self.sn_caculator.state_dict() if exists(self.sn_caculator) else None,
                'opt_d' : self.opt_d.state_dict() if exists(self.opt_d) else None,
                'criterion_2d' : self.accelerator.get_state_dict(self.criterion) if exists(self.criterion) else None,
                }
        torch.save(data, os.path.join(self.results_folder, 'model-{}.pt'.format(step)))
        torch.save(data, os.path.join(self.results_folder, 'model-last.pt'.format(step)))
 
    def load(self, pth):
        data = torch.load(pth, map_location = 'cpu')
        self.model.load_state_dict(data['model'])
        self.mlp.load_state_dict(data['mlp'])
        self.step = data['step']
        self.current_iters = data['current_iters']
        self.opt.load_state_dict(data['opt'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

        if exists(self.scheduler) and exists(data['opt_sched']):
            self.scheduler.load_state_dict(data['opt_sched'])
        
        if exists(self.sn_caculator) and data['sn'] is not None:
            self.sn_caculator.load_state_dict(data['sn'], device = self.accelerator.device)
        
        try:
            self.opt_d.load_state_dict(data['opt_d'])
            self.criterion.load_state_dict(data['criterion_2d'])
        except:
            print('Not found criterion!')
            pass

    def train(self):
        device = self.accelerator.device
        optimizer_idx = True

        with tqdm(initial = self.step, total = self.epochs) as pbar:
            while self.step < self.epochs:
                if self.step < self.warmup_epochs:
                    lr = self.train_lr * float(self.step + 1) / self.warmup_epochs
                    for param_group in self.opt.param_groups:
                        param_group['lr'] = lr

                if self.step > self.warmup_epochs and self.scheduler is not None:
                    self.scheduler.step()

                self.model.train()
                self.mlp.train()
                total_losses = 0.
        
                for idx, (x, _) in enumerate(self.data):
                    #x, _ = next(self.data)
                    assert len(x.shape) == 4
                    x = symmetrize_image_data(x) # 512x512
                    ## Resize image
                    target, coords, scale, y = multiscale_image_transform(x, self.args.resolution, self.multiscale, device)
                    with self.accelerator.autocast():
                        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                            posterior = self.model.module.encode(y)
                            pe = self.model.module.decode(posterior.sample())
                        else:
                            posterior = self.model.encode(y)
                            pe = self.model.decode(posterior.sample())
                        output = self.mlp(coords, hdbf = pe, si = scale)
                        if optimizer_idx:
                            ## KL loss
                            kld = posterior.kl(mean=False)
                            kld_loss = torch.mean(kld)
                            if self.kl_anneal:
                                kl_coeff = linear_kl_coeff(self.current_iters, self.kl_anneal_portion * self.num_total_iters,
                                                            self.kl_const_portion * self.num_total_iters, self.kl_const_coeff, self.kl_max_coeff)
                            else:
                                kl_coeff = self.kl_max_coeff

                            ## Recon Loss
                            recon_loss = torch.sum(torch.abs(output.contiguous() - target.contiguous()), dim = (1,2,3))        
                            recon_loss = torch.mean(recon_loss)
                            #recon_loss = torch.mean(torch.abs(output.contiguous() - target.contiguous()))        

                            ## Perceptual loss
                            p_coeff = 1.
                            p_loss = self.perceptual_loss(target.contiguous(), output.contiguous()).mean()

                            total_loss = recon_loss + kl_coeff * kld_loss + p_coeff * p_loss
                            
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

                        ## Patchwise adversarial loss
                        if self.args.loss_config.adversarial:
                            gan_loss = self.criterion(inputs=target, reconstructions=output, optimizer_idx=optimizer_idx, cond=scale)
                            total_loss += gan_loss
                        else:
                            gan_loss = 0.
                        total_loss = total_loss / self.gradient_accumulate_every
                    pbar.set_description('recon : {:.4f} \ kl : {:.4f} \ g_loss : {:.4f}'.format(recon_loss, kld_loss.item(), gan_loss))
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
                            #self.accelerator.clip_grad_norm_(self.criterion.module.discriminator_2d.parameters(), self.max_grad_norm)
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
                    if self.test_data is not None:
                        coords = convert_to_coord_format_2d(1, 256, 256, device = device, hstart=-255/256, hend = 255/256, wstart=-255/256, wend = 255/256)
                        fid = test_rfid(self.model, self.mlp, coords, self.test_data, self.results_pth, device, save = False)
                        print('FID:', fid)
                    else:
                        pass

                    vtils.save_image(output, os.path.join(self.results_pth, '{}.png'.format(self.step)), normalize=True, scale_each=True)
                    self.save(step = self.step)

                self.accelerator.wait_for_everyone()
                self.step += 1
                pbar.update(1)
    
    def eval(self):
        device = self.accelerator.device
        coords = convert_to_coord_format_2d(1, self.test_resolution, 
                                            self.test_resolution, 
                                            device = device, 
                                            hstart=-(self.test_resolution-1)/self.test_resolution, 
                                            hend = (self.test_resolution-1)/self.test_resolution, 
                                            wstart=-(self.test_resolution-1)/self.test_resolution, 
                                            wend = (self.test_resolution-1)/self.test_resolution)
        
        if self.test_data is not None:
            fid = test_rfid(self.model, self.mlp, coords, self.test_data, self.results_pth, device, save = True)
            print('FID:', fid)
        else:
            pass
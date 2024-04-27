import torch
#from torch.utils import data
from torchvision import transforms
import torchvision.datasets as dsets
from utils.videoloader import get_loaders


def first_stage_train(args):
    ## Image
    if args.domain == 'image':
        from tools.d2c_vae.image import D2CTrainer
        from models.d2c_vae.autoencoder_unet import Autoencoder
        from models.d2c_vae.mlp import MLP
        from losses.perceptual import LPIPSWithDiscriminator2D

        ## Get data
        transform_list = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform_list = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((args.data_config.test_resolution, args.data_config.test_resolution)),
            transforms.ToTensor(),
        ])
        train_data = dsets.ImageFolder(args.data_config.data_dir, transform=transform_list)
        train_loader = torch.utils.data.DataLoader(train_data, 
                                       batch_size=args.data_config.batch_size,
                                       shuffle=True,
                                       num_workers=4,
                                       pin_memory=True,
                                       drop_last=True)
        
        test_data = dsets.ImageFolder(args.data_config.test_data_dir, transform=test_transform_list)
        test_loader = torch.utils.data.DataLoader(test_data, 
                                       batch_size=args.data_config.test_batch_size,
                                       shuffle=False,
                                       num_workers=2,
                                       pin_memory=False,
                                       drop_last=False)

        ## Get model
        vaemodel = Autoencoder(ddconfig=args.ddconfig, embed_dim=args.embed_dim)
        mlp = MLP(**args.mlpconfig)
        if args.loss_config.adversarial:
            criterion = LPIPSWithDiscriminator2D(disc_in_channels=4, disc_weight=args.loss_config.disc_weight)
        else:
            criterion = None

        ## Get trainer
        trainer = D2CTrainer(args, vaemodel, mlp, train_loader, test_loader, criterion)

    ## 3D (occupancy)
    elif args.domain == 'occupancy':
        from tools.d2c_vae.occupancy import D2CTrainer
        from models.d2c_vae.autoencoder_unet import Autoencoder3D
        from models.d2c_vae.mlp import MLP3D as MLP
        from models.d2c_vae.pointnet import LocalPoolPointnet
        from convocc.src import data, config

        ## Get data
        cfg = config.load_config(args.data_config.conv_config, 'convocc/configs/default.yaml')
        train_data = config.get_dataset('train', cfg)
        train_loader = torch.utils.data.DataLoader(train_data,
                                       batch_size = args.data_config.batch_size,
                                       shuffle=True,
                                       num_workers=4,
                                       collate_fn=data.collate_remove_none,
                                       worker_init_fn=data.worker_init_fn
                                       )
        
        ## Get model
        # pointnet/config from https://github.com/autonomousvision/convolutional_occupancy_networks
        pointnet = LocalPoolPointnet(dim=cfg['data']['dim'], c_dim=cfg['model']['c_dim'], padding=cfg['data']['padding'], **cfg['model']['encoder_kwargs'])
        vaemodel = Autoencoder3D(ddconfig=args.ddconfig, embed_dim=args.embed_dim)
        mlp = MLP(**args.mlpconfig)

        ## Mesh generator
        mesh_gen = config.get_generator(cfg)

        ## Get trainer
        trainer = D2CTrainer(args, pointnet, vaemodel, mlp, mesh_gen, data=train_loader)
    
    ## Video
    elif args.domain == 'video':
        from tools.d2c_vae.video import D2CTrainer
        from models.d2c_vae.autoencoder_vit import VITAutoencoder
        from models.d2c_vae.mlp import MLPVideo as MLP
        from losses.perceptual import LPIPSWithDiscriminator3D

        ## Get data
        train_loader, test_loader = get_loaders(0, args.data_config.data_dir, 
                                                args.data_config.dataset,
                                                resolution=args.resolution,
                                                timesteps=args.data_config.frames, 
                                                skip=1, 
                                                batch_size=args.data_config.batch_size,
                                                test_batch_size=args.data_config.test_batch_size)

        ## Get model
        vaemodel = VITAutoencoder(ddconfig=args.ddconfig, embed_dim = args.embed_dim, frames=args.data_config.frames)
        mlp = MLP(**args.mlpconfig)
        if args.loss_config.adversarial:
            criterion = LPIPSWithDiscriminator3D(disc_weight=args.loss_config.disc_weight, timesteps=args.data_config.frames)
        else:
            criterion = None

        ## Get trainer
        trainer = D2CTrainer(args, vaemodel, mlp, data=train_loader, test_data=test_loader, criterion=criterion)

    ## NeRF
    elif args.domain =='nerf':
        from tools.d2c_vae.nerf import D2CTrainer
        from models.d2c_vae.autoencoder_unet import Autoencoder
        from models.d2c_vae.mlp import MLPNeRF as MLP
        from models.d2c_vae.pointnet import LocalPoolPointnet
        from convocc.src import data, config
        from utils import nerf_helpers
        from utils.nerf_dataset import NeRFShapeNetDataset

        ## Get data
        cfg = config.load_config(args.data_config.conv_config, 'convocc/configs/default.yaml')
        train_data = NeRFShapeNetDataset(root_dir=cfg['data']['path'], classes=cfg['data']['classes'])
        train_loader = torch.utils.data.DataLoader(train_data,
                                       batch_size = args.data_config.batch_size,
                                       shuffle=True,
                                       num_workers=4,
                                       )
        
        embed_fn, cfg['model']['TN']['input_ch_embed'] = nerf_helpers.get_embedder(cfg['model']['TN']['multires'], cfg['model']['TN']['i_embed'])
        embeddirs_fn, cfg['model']['TN']['input_ch_views_embed']= nerf_helpers.get_embedder(cfg['model']['TN']['multires_views'], cfg['model']['TN']['i_embed'])
        
        ## Get model
        # pointnet/config from https://github.com/autonomousvision/convolutional_occupancy_networks
        pointnet = LocalPoolPointnet(dim=cfg['data']['dim'], c_dim=cfg['model']['c_dim'], padding=cfg['data']['padding'], **cfg['model']['encoder_kwargs'])
        vaemodel = Autoencoder(ddconfig=args.ddconfig, embed_dim=args.embed_dim)
        mlp = MLP(**args.mlpconfig, in_channels_dir=cfg['model']['TN']['input_ch_views_embed'])

        ## Mesh generator
        mesh_gen = config.get_generator(cfg)

        ## Get trainer
        trainer = D2CTrainer(args, pointnet, vaemodel, mlp, mesh_gen, embed_fn, embeddirs_fn, cfg, data=train_loader)
    
    else:
        raise ValueError('Undefined Domain!')
    

    if args.mode == 'train':
        ## Train
        trainer.train()
        trainer.save()
    elif args.mode == 'eval':
        trainer.eval()
    else:
        raise ValueError
    

def second_stage_train(args):
    if args.domain == 'image':
        from tools.ldm.image import LDMTrainer
        from models.d2c_vae.autoencoder_unet import Autoencoder
        from models.d2c_vae.mlp import MLP
        from diffusion.ddpm import DDPM

        ## Get data
        transform_list = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        test_transform_list = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((args.data_config.test_resolution, args.data_config.test_resolution)),
            transforms.ToTensor(),
        ])

        train_data = dsets.ImageFolder(args.data_config.data_dir, transform=transform_list)
        train_loader = torch.utils.data.DataLoader(train_data, 
                                       batch_size=args.data_config.batch_size,
                                       shuffle=True,
                                       num_workers=4,
                                       pin_memory=True,
                                       drop_last=True)

        test_data = dsets.ImageFolder(args.data_config.test_data_dir, transform=test_transform_list)
        test_loader = torch.utils.data.DataLoader(test_data, 
                                       batch_size=args.data_config.test_batch_size,
                                       shuffle=False,
                                       num_workers=2,
                                       pin_memory=False,
                                       drop_last=False)
        ## Get model
        vaemodel = Autoencoder(ddconfig=args.ddconfig, embed_dim=args.embed_dim)
        mlp = MLP(**args.mlpconfig)
        if args.DiT:
            from models.ldm.modules.diffusionmodules.maskedtransformer import MDTv2
            # MDTv2-L
            diffusionmodel = MDTv2(input_size=64, in_channels=64, depth=24, hidden_size=1024, patch_size=2, num_heads=16,)
        else:
            from models.ldm.modules.diffusionmodules.openaimodel import UNetModel
            diffusionmodel = UNetModel(**args.unetconfig)
        diffusion_process = DDPM(model=diffusionmodel, **args.ddpmconfig)

        ## Get trainer
        trainer = LDMTrainer(args, vaemodel, mlp, diffusionmodel, diffusion_process, train_loader, test_loader)

    elif args.domain == 'occupancy':
        from tools.ldm.occupancy import LDMTrainer
        from models.d2c_vae.autoencoder_unet import Autoencoder3D
        from models.d2c_vae.mlp import MLP3D as MLP
        from models.d2c_vae.pointnet import LocalPoolPointnet
        from convocc.src import data, config
        from diffusion.ddpm import DDPM

        ## Get data
        cfg = config.load_config(args.data_config.conv_config, 'convocc/configs/default.yaml')
        train_data = config.get_dataset('train', cfg)
        train_loader = torch.utils.data.DataLoader(train_data,
                                       batch_size = args.data_config.batch_size,
                                       shuffle=True,
                                       num_workers=4,
                                       collate_fn=data.collate_remove_none,
                                       worker_init_fn=data.worker_init_fn
                                       )
        
        ## Get model
        # pointnet/config from https://github.com/autonomousvision/convolutional_occupancy_networks
        pointnet = LocalPoolPointnet(dim=cfg['data']['dim'], c_dim=cfg['model']['c_dim'], padding=cfg['data']['padding'], **cfg['model']['encoder_kwargs'])
        vaemodel = Autoencoder3D(ddconfig=args.ddconfig, embed_dim=args.embed_dim)
        mlp = MLP(**args.mlpconfig)
        if args.DiT:
            from models.ldm.modules.diffusionmodules.maskedtransformer import MDTv2
            diffusionmodel = MDTv2(input_size=16, in_channels=4*3, depth=24, hidden_size=1024, patch_size=2, num_heads=16, mask_ratio=0.3, cross_plane=False)
        else:
            from models.ldm.modules.diffusionmodules.openaimodel import UNetModel
            diffusionmodel = UNetModel(**args.unetconfig)
        diffusion_process = DDPM(model=diffusionmodel, **args.ddpmconfig)

        ## Get Mesh generator
        mesh_gen = config.get_generator(cfg)
        ## Get trainer
        trainer = LDMTrainer(args, pointnet, vaemodel, mlp, diffusionmodel, diffusion_process, train_loader, mesh_gen)

    elif args.domain == 'video':
        from tools.ldm.video import LDMTrainer
        from models.d2c_vae.autoencoder_vit import VITAutoencoder
        from models.d2c_vae.mlp import MLPVideo as MLP
        from diffusion.ddpm import DDPM

        ## Get data
        train_loader, test_loader = get_loaders(0, args.data_config.data_dir, 
                                                args.data_config.dataset,
                                                resolution=args.resolution,
                                                timesteps=args.data_config.frames, 
                                                skip=1, 
                                                batch_size=args.data_config.batch_size,
                                                test_batch_size=args.data_config.test_batch_size)

        ## Get model
        vaemodel = VITAutoencoder(ddconfig=args.ddconfig, embed_dim = args.embed_dim, frames=args.data_config.frames)
        mlp = MLP(**args.mlpconfig)
        if args.DiT:
            raise NotImplementedError
        else:
            from models.ldm.modules.diffusionmodules.openaimodel import UNetModel_Triplane
            diffusionmodel = UNetModel_Triplane(**args.unetconfig)

        ## Get trainer
        trainer = LDMTrainer(args, vaemodel, mlp, diffusionmodel, diffusion_process, train_loader, test_loader)

    else:
        raise ValueError('Undefined Domain!')
    

    if args.mode == 'train':
        ## Train
        trainer.train()
        trainer.save()
    elif args.mode == 'eval':
        print('Evaluation!')
        trainer.eval()
    else:
        raise ValueError

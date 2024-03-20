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
        raise NotImplementedError
    
    ## Video
    elif args.domain == 'video':
        raise NotImplementedError

    ## NeRF
    elif args.domain =='nerf':
        raise NotImplementedError
    
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
        from models.ldm.modules.diffusionmodules.openaimodel import UNetModel
        from diffusion.ddpm import DDPM

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
        diffusionmodel = UNetModel(**args.unetconfig)
        diffusion_process = DDPM(model=diffusionmodel, **args.ddpmconfig)

        ## Get trainer
        trainer = LDMTrainer(args, vaemodel, mlp, diffusionmodel, diffusion_process, train_loader, test_loader)

    elif args.domain == 'occupancy':
        raise NotImplementedError

    elif args.domain == 'video':
        raise NotImplementedError

    elif args.domain == 'nerf':
        raise NotImplementedError
    
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

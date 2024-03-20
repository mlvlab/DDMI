
import os
import argparse

from omegaconf import OmegaConf

from utils.general_utils import random_seed
from exp.stage import first_stage_train, second_stage_train


def main(args):
    if args.exp == 'd2c-vae':
        config = OmegaConf.load(args.configs)
        args.data_config = config.data
        args.ddconfig = config.model.params.ddconfig
        args.mlpconfig = config.model.params.mlpconfig
        args.loss_config = config.model.params.lossconfig
        args.embed_dim = config.model.embed_dim
        args.lr = config.model.lr
        args.resolution = config.model.params.ddconfig.resolution
        args.resume = config.model.resume
        args.use_fp16 = config.model.use_fp16
        args.amp = config.model.amp
        args.domain = config.data.domain
        args.mode = config.data.mode

        first_stage_train(args)

    elif args.exp == 'ldm':
        config = OmegaConf.load(args.configs)
        args.data_config = config.data
        args.ddconfig = config.model.params.ddconfig
        args.mlpconfig = config.model.params.mlpconfig
        args.unetconfig = config.model.params.unetconfig
        args.loss_config = config.model.params.lossconfig
        args.ddpmconfig = config.model.params.ddpmconfig
        args.embed_dim = config.model.embed_dim
        args.lr = config.model.lr
        args.resolution = config.model.params.ddconfig.resolution
        args.resume = config.model.resume
        args.amp = config.model.amp
        args.use_fp16 = config.model.use_fp16
        args.domain = config.data.domain
        args.mode = config.data.mode

        second_stage_train(args)

    else:
        raise ValueError('Undefined Type!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, required=True, choices=['d2c-vae', 'ldm'])
    parser.add_argument('--configs', type=str)
    parser.add_argument('--seed', type=int, default=777)

    args = parser.parse_args()

    # seed
    random_seed(args.seed)

    main(args)

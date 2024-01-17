"""
Train a diffusion model on images.
"""

import argparse
import yaml

from cc import dist_util, logger
from cc.image_datasets import load_data, load_lmdb, load_npy
from cc.resample import create_named_schedule_sampler
from cc.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    cc_train_defaults,
    args_to_dict,
    add_dict_to_argparser,
)
from cc.train_util import CCTrainLoop
from inverse.measurements import get_noise, get_operator
import torch.distributed as dist


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(args=args)

    # # print args
    # logger.log("args:")
    # for arg in vars(args):
    #     logger.log(f"{arg}: {getattr(args, arg)}")

    logger.log("creating model and diffusion...")
    model_and_diffusion_kwargs = args_to_dict(
        args, model_and_diffusion_defaults().keys()
    )
    model_and_diffusion_kwargs["distillation"] = True
    model_and_diffusion_kwargs["control"] = True
    control_net, controlled_unet, diffusion = create_model_and_diffusion(**model_and_diffusion_kwargs)
    
    if len(args.unet_path) > 0:
        logger.log(f"loading control net from {args.unet_path}")
        control_net.load_state_dict(
            dist_util.load_state_dict(args.unet_path, map_location="cpu"), 
            strict=False
        )
        logger.log(f"loading controlled unet from {args.unet_path}")
        controlled_unet.load_state_dict(
            dist_util.load_state_dict(args.unet_path, map_location="cpu")
        )
    control_net.to(dist_util.dev())
    control_net.train()
    controlled_unet.to(dist_util.dev())
    controlled_unet.train()

    if args.use_fp16:
        control_net.convert_to_fp16()
        controlled_unet.convert_to_fp16()

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    if args.batch_size == -1:
        batch_size = args.global_batch_size // dist.get_world_size()
        if args.global_batch_size % dist.get_world_size() != 0:
            logger.log(
                f"warning, using smaller global_batch_size of {dist.get_world_size()*batch_size} instead of {args.global_batch_size}"
            )
    else:
        batch_size = args.batch_size

    if args.dataset_mode == "image_folder":
        data = load_data(
            data_dir=args.data_dir,
            batch_size=batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
        )
    elif args.dataset_mode == "lmdb":
        data = load_lmdb(
            data_dir=args.data_dir,
            batch_size=batch_size,
            image_size=args.image_size,
        )
    elif args.dataset_mode == "npy":
        data = load_npy(
            data_dir=args.data_dir,
            batch_size=batch_size,
        )
    else:
        raise ValueError(f"unknown dataset mode {args.dataset_mode}")

    if len(args.task_config) > 0:
        logger.log("creating task...")
        # Load task configurations
        task_config = load_yaml(args.task_config)
        # Prepare Operator and noise
        measure_config = task_config['measurement']
        noiser = get_noise(**measure_config['noise'])
        operator = get_operator(device=dist_util.dev(), **measure_config['operator'])
        assert noiser is not None, "Noise should be determined if given the operator!"
        if measure_config['operator']['name'] == 'ct':
            condition_operator = lambda x: operator.recon(x)
        elif measure_config['operator']['name'] == 'nonlinear_blur':
            condition_operator = lambda x: noiser(operator.forward(x))
        else:
            condition_operator = lambda x: operator.transpose(noiser(operator.forward(x)))
        logger.log(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    logger.log("training...")
    CCTrainLoop(
        model=control_net,
        controlled_unet=controlled_unet,
        total_training_steps=args.total_training_steps,
        diffusion=diffusion,
        data=data,
        batch_size=batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        condition_operator=condition_operator,
        noiser=noiser,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        dataset_mode="image_folder", 
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        global_batch_size=2048,
        batch_size=-1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=5000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        task_config="",
        wandb_api_key="",
        wandb_user="", 
        name="",
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(cc_train_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

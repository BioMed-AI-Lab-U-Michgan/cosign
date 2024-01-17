"""
Train a diffusion model on images.
"""

import argparse

from cc import dist_util, logger
from cc.image_datasets import load_data, load_npy
from cc.resample import create_named_schedule_sampler
from cc.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from cc.train_util import TrainLoop
import torch.distributed as dist


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(args=args)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    if len(args.pretrained_model_path) > 0:
        logger.log(f"loading the pretrained model from {args.pretrained_model_path}")
        state_dict = dist_util.load_state_dict(args.pretrained_model_path, map_location="cpu")
        if args.in_channels != 3:
            state_dict.pop("input_blocks.0.0.weight")
            state_dict.pop("out.2.weight")
            state_dict.pop("out.2.bias")
        model.load_state_dict(
            state_dict,
            strict=False,
        )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()

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
    elif args.dataset_mode == "npy":
        data = load_npy(
            data_dir=args.data_dir,
            batch_size=batch_size,
        )

    logger.log("creating data loader...")

    logger.log("training...")
    TrainLoop(
        model=model,
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
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        pretrained_model_path="",
        wandb_api_key="",
        wandb_user="", 
        name="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import yaml

import numpy as np
import torch as th
import torch.distributed as dist
import torchvision.transforms as transforms

from cc import dist_util, logger
from cc.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from cc.random_util import get_generator
from cc.karras_diffusion import karras_sample, control_sample
from inverse.measurements import get_noise, get_operator, LinearOperator
from inverse.condition_methods import get_conditioning_method
from cc.valid_datasets import get_dataset, get_dataloader


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
def main():
    args = create_argparser().parse_args()

    # # print args
    # logger.log("args:")
    # for arg in vars(args):
    #     logger.log(f"{arg}: {getattr(args, arg)}")

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model_and_diffusion_kwargs = args_to_dict(
        args, model_and_diffusion_defaults().keys()
    )
    model_and_diffusion_kwargs["distillation"] = True
    model_and_diffusion_kwargs["control"] = True
    control_net, controlled_unet, diffusion = create_model_and_diffusion(**model_and_diffusion_kwargs)
    control_net.load_state_dict(
            dist_util.load_state_dict(args.control_net_path, map_location="cpu"),
        )
    control_net.to(dist_util.dev())
    control_net.eval()
    controlled_unet.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu"),
    )
    controlled_unet.to(dist_util.dev())
    controlled_unet.eval()
    if args.use_fp16:
        control_net.convert_to_fp16()
        controlled_unet.convert_to_fp16()
    model = controlled_unet

    if len(args.task_config) > 0:
        logger.log("creating task...")
        # Load task configurations
        task_config = load_yaml(args.task_config)

        # Prepare Operator and noise
        measure_config = task_config['measurement']
        operator = get_operator(device=dist_util.dev(), **measure_config['operator'])
        noiser = get_noise(**measure_config['noise'])
        logger.log(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

        # Prepare conditioning method
        cond_config = task_config['conditioning']
        params = cond_config['params'] if cond_config['params'] is not None else {}
        cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **params)
        measurement_cond_fn = cond_method.conditioning
        print(f"Conditioning method : {task_config['conditioning']['method']}")

        # Prepare dataloader
        data_config = task_config['data']
        if "ldct" in data_config['name']:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5, ), (0.5, ))])
        else:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = get_dataset(**data_config, transforms=transform)
        loader = get_dataloader(dataset, batch_size=args.batch_size, num_workers=0, train=False)

        condition_args = dict(
            measurement_cond_fn=measurement_cond_fn
        )
    else:
        condition_args = None

    logger.log("sampling...")
    if args.sampler in ["multistep", "ddnm"]:
        assert len(args.ts) > 0
        ts = tuple(int(x) for x in args.ts.split(","))
    else:
        ts = None

    all_images, all_conds = [], []
    all_labels = []
    all_output = {'uncond_x0':[], \
                  'cond_x0':[], \
                  'uncond_xt_next':[], \
                  'cond_xt_next':[]}
    intermediates = {}
    generator = get_generator(args.generator, args.num_samples, args.seed)

    for ref_img in loader:
        # Forward measurement model
        y = operator.forward(ref_img.to(dist_util.dev()))
        y_n = noiser(y)
        condition_args['y_n'] = y_n
        if task_config['measurement']['operator']['name'] != 'ct':
            if isinstance(operator, LinearOperator) is True:
                condition_args['hint'] = operator.transpose(y_n)
            else:
                condition_args['hint'] = y_n
        else:
            y_n = operator.recon(ref_img)   # only for saving purpose
            condition_args['hint'] = y_n
        
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes

        sample, intermediates = control_sample(
            diffusion,
            control_net,
            controlled_unet,
            (args.batch_size, args.in_channels, args.image_size, args.image_size),
            steps=args.steps,
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
            clip_denoised=args.clip_denoised,
            sampler=args.sampler,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            s_churn=args.s_churn,
            s_tmin=args.s_tmin,
            s_tmax=args.s_tmax,
            s_noise=args.s_noise,
            generator=generator,
            ts=ts,
            condition_args=condition_args,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

        if 'images' in intermediates:
            y_n = intermediates['images']
        y_n = ((y_n + 1) * 127.5).clamp(0, 255).to(th.uint8)
        y_n = y_n.permute(0, 2, 3, 1)
        y_n = y_n.contiguous()
        gathered_y = [th.zeros_like(y_n) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_y, y_n)  # gather not supported with NCCL
        all_conds.extend([y_n.cpu().numpy() for y_n in gathered_y])

        for name in ['uncond_x0', 'cond_x0', 'uncond_xt_next', 'cond_xt_next']:
            if name in intermediates:
                intermediate = intermediates[name]
                intermediate = ((intermediate + 1) * 127.5).clamp(0, 255).to(th.uint8)
                intermediate = intermediate.permute(1, 0, 3, 4, 2)
                intermediate = intermediate.contiguous()
                gathered_intermediate = [th.zeros_like(intermediate) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_intermediate, intermediate)  # gather not supported with NCCL
                all_output[name].extend([inter.cpu().numpy() for inter in gathered_intermediate])

        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        # logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    # condition
    orig_arr = np.concatenate(all_conds, axis=0)
    orig_arr = orig_arr[: args.num_samples]
    # intermediates
    for name in ['uncond_x0', 'cond_x0', 'uncond_xt_next', 'cond_xt_next']:
        if name in intermediates:
            inter_arr = np.concatenate(all_output[name], axis=0)
            all_output[name] = inter_arr[: args.num_samples] #[B, N, 3, size, size]
    
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(
            out_path, 
            generated = arr, 
            original = orig_arr,
            label_arr = label_arr if args.class_cond else np.array([]),
            uncond_x0 = all_output['uncond_x0'] if len(all_output['uncond_x0'])>0 else np.array([]),
            cond_x0 = all_output['cond_x0'] if len(all_output['cond_x0'])>0 else np.array([]),
            uncond_xt_next = all_output['uncond_xt_next'] if len(all_output['uncond_xt_next'])>0 else np.array([]),
            cond_xt_next = all_output['cond_xt_next'] if len(all_output['cond_xt_next'])>0 else np.array([])
        )

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        training_mode="edm",
        generator="determ",
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        sampler="heun",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        steps=40,
        model_path="",
        control_net_path="",
        seed=42,
        ts="",
        task_config=""
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

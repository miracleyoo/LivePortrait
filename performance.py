# coding: utf-8
"""
for humankind
"""

import os
import os.path as osp
import tyro
import torch
import subprocess
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.live_portrait_pipeline import LivePortraitPipeline
from fvcore.nn import FlopCountAnalysis
from functools import partial


def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False


def fast_check_args(args: ArgumentConfig):
    if not osp.exists(args.source):
        raise FileNotFoundError(f"source info not found: {args.source}")
    if not osp.exists(args.driving):
        raise FileNotFoundError(f"driving info not found: {args.driving}")


def main():
    # set tyro theme
    tyro.extras.set_accent_color("bright_cyan")
    args = tyro.cli(ArgumentConfig)

    ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg")
    if osp.exists(ffmpeg_dir):
        os.environ["PATH"] += (os.pathsep + ffmpeg_dir)

    if not fast_check_ffmpeg():
        raise ImportError(
            "FFmpeg is not installed. Please install FFmpeg (including ffmpeg and ffprobe) before running this script. https://ffmpeg.org/download.html"
        )

    fast_check_args(args)

    # specify configs for inference
    inference_cfg = partial_fields(InferenceConfig, args.__dict__)
    crop_cfg = partial_fields(CropConfig, args.__dict__)

    live_portrait_pipeline = LivePortraitPipeline(
        inference_cfg=inference_cfg,
        crop_cfg=crop_cfg
    )

    # run
    # live_portrait_pipeline.execute(args)

    # Performance Check
    model_wrapper = live_portrait_pipeline.live_portrait_wrapper
    model_f = model_wrapper.appearance_feature_extractor
    model_m = model_wrapper.motion_extractor
    model_w = model_wrapper.warping_module
    model_g = model_wrapper.spade_generator
    model_s = model_wrapper.stitching_retargeting_module
    models = {'f':model_f, 'm':model_m, 'w':model_w, 'g':model_g, 's':model_s['stitching']}

    inputs = {
        'f': torch.rand(1, 3, 256, 256),
        'm': torch.rand(1, 3, 256, 256),
        'w': torch.rand(1, 32, 16, 64, 64),
        'g': torch.rand(1, 256, 64, 64),
        's': torch.rand(1, 126)
    }

    # face image output shape: (512, 512, 3)
    # full image (s9) output shape: (1280, 720, 3)

    # Perform the FLOP analysis
    flops_total = 0
    param_total = 0
    for mname, m in models.items():
        input_tensor = inputs[mname].cuda()
        if mname == 'w':
            m.forward = partial(m.forward, kp_source=torch.rand(1, 21, 3).cuda(), kp_driving=torch.rand(1, 21, 3).cuda())
            flops = FlopCountAnalysis(m, input_tensor)
        else:
            flops = FlopCountAnalysis(m, input_tensor)
        flops_total += flops.total()
        print(f"{mname} FLOPs: {flops.total() / 1e9:.3f} GFLOPs")
        param_num = sum(p.numel() for p in m.parameters())
        param_total += param_num
        print(f'{mname} parameter number: {param_num}')

    print(f"Total FLOPs: {flops_total / 1e9:.3f} GFLOPs")
    print(f"Total Parameter Number: {param_total}")


if __name__ == "__main__":
    main()

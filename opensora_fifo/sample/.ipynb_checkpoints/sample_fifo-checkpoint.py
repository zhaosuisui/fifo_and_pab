import math
import os
import torch
import argparse
import torchvision

from diffusers.schedulers import (DDIMScheduler, DDPMScheduler, PNDMScheduler,
                                  EulerDiscreteScheduler, DPMSolverMultistepScheduler,
                                  HeunDiscreteScheduler, EulerAncestralDiscreteScheduler,
                                  DEISMultistepScheduler, KDPM2AncestralDiscreteScheduler)
from diffusers.schedulers.scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler
from diffusers.models import AutoencoderKL, AutoencoderKLTemporalDecoder
from omegaconf import OmegaConf
from torchvision.utils import save_image
from transformers import T5EncoderModel, T5Tokenizer, AutoTokenizer

import os, sys
from tqdm import trange, tqdm

sys.path.append(os.path.split(os.path.split(sys.path[0])[0])[0])
from opensora_fifo.models.diffusion.latte.modeling_latte import LatteT2V
from opensora.models.ae import ae_stride_config, getae, getae_wrapper
from opensora.models.ae.videobase import CausalVQVAEModelWrapper, CausalVAEModelWrapper
from opensora.models.text_encoder import get_text_enc
from opensora.utils.utils import save_video_grid

sys.path.append(os.path.split(sys.path[0])[0])
from pipeline_videogen import VideoGenPipeline

import imageio
import copy

def prepare_latents(args, latents_dir, scheduler):
    latents_list = []
    video = torch.load(os.path.join(latents_dir, "video.pt"))

    timesteps = scheduler.timesteps
    # 前瞻预测
    if args.lookahead_denoising:
        # 生成前半部分帧的潜在表示，前半部分帧被处理为含有大量噪声的潜在表示，作为未来帧，为后续的去噪做准备
        for i in range(args.video_length // 2):
            # 取最后一个时间步（噪声较大）
            t = timesteps[-1]
            alpha = scheduler.alphas_cumprod[t]
            beta = 1 - alpha
            x_0 = video[:,:,[0]]
            # 这里的潜在表示就是原始帧+噪声？
            latents = alpha**(0.5) * x_0 + beta**(0.5) * torch.randn_like(x_0)
            latents_list.append(latents)
        # 对队列中的帧依次生成潜在表示
        for i in range(args.queue_length):
            t = timesteps[args.queue_length-i-1]
            alpha = scheduler.alphas_cumprod[t]
            beta = 1 - alpha
            frame_idx = max(0, i-(args.queue_length - args.video_length))
            x_0 = video[:,:,[frame_idx]]
            
            latents = alpha**(0.5) * x_0 + beta**(0.5) * torch.randn_like(x_0)
            latents_list.append(latents)
    else:
        for i in range(args.queue_length):
            t = timesteps[args.queue_length-i-1]
            alpha = scheduler.alphas_cumprod[t]
            beta = 1 - alpha

            frame_idx = max(0, i-(args.queue_length - args.video_length))
            x_0 = video[:,:,[frame_idx]]
            
            latents = alpha**(0.5) * x_0 + beta**(0.5) * torch.randn_like(x_0)
            latents_list.append(latents)
    # 把列表中的潜在表示在第二个维度（高或宽）拼接起来，得到最终的潜在表示（包含所有信息），可以作为模型的输入
    latents = torch.cat(latents_list, dim=2)
    return latents

def shift_latents(latents, scheduler):
    # 平移更新潜在表示，全体向左移动一帧，是不是第一帧出队？
    latents[:,:,:-1] = latents[:,:,1:].clone()
    # 给最后一帧添加新的噪声
    # add new noise to the last frame
    latents[:,:,-1] = torch.randn_like(latents[:,:,-1]) * scheduler.init_noise_sigma

    return latents


def main(args):
    # torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load VAE:
    vae = getae_wrapper(args.ae)(args.model_path, subfolder="vae", cache_dir=args.cache_dir).to(device, dtype=torch.float16)
    # 是否启用切片，将大数据集分为一块一块处理
    if args.enable_tiling:
        vae.vae.enable_tiling()
        vae.vae.tile_overlap_factor = args.tile_overlap_factor
    vae.vae_scale_factor = ae_stride_config[args.ae]
    # Load model:
    # 加载模型
    transformer_model = LatteT2V.from_pretrained(args.model_path, subfolder=args.version, cache_dir=args.cache_dir, torch_dtype=torch.float16).to(device)
    transformer_model.force_images = args.force_images
    tokenizer = T5Tokenizer.from_pretrained(args.text_encoder_name, cache_dir=args.cache_dir)
    text_encoder = T5EncoderModel.from_pretrained(args.text_encoder_name, cache_dir=args.cache_dir, torch_dtype=torch.float16).to(device)

    # video_length, image_size = transformer_model.config.video_length, int(args.version.split('x')[1])
    # latent_size = (image_size // ae_stride_config[args.ae][1], image_size // ae_stride_config[args.ae][2])
    # vae.latent_size = latent_size
    # 也可以用该模型生成图像
    if args.force_images:
        video_length = 1
        ext = 'jpg'
    else:
        ext = 'mp4'

    # set eval mode
    transformer_model.eval()
    vae.eval()
    text_encoder.eval()
    # 根据用户输入的采样方法选择调度器，调度器用来控制采样步骤，控制在生成过程不同时间步噪声的引入与去除
    # 采样在训练之后，训练向图像加噪，学习还原回原图，采样是从正态分布采样噪声，去噪生成新图片
    schedulers = None
    if args.sample_method == 'DDIM':  #########
        scheduler = DDIMScheduler()
    elif args.sample_method == 'EulerDiscrete':
        scheduler = EulerDiscreteScheduler()
    elif args.sample_method == 'DDPM':  #############
        scheduler = DDPMScheduler()
    elif args.sample_method == 'DPMSolverMultistep':
        scheduler = DPMSolverMultistepScheduler()
    elif args.sample_method == 'DPMSolverSinglestep':
        scheduler = DPMSolverSinglestepScheduler()
    elif args.sample_method == 'PNDM':
        scheduler = PNDMScheduler()
        schedulers = [PNDMScheduler() for _ in range(args.video_length)]
        for s in schedulers:
            s.set_timesteps(args.num_sampling_steps, device=device)
    elif args.sample_method == 'HeunDiscrete':  ########
        scheduler = HeunDiscreteScheduler()
    elif args.sample_method == 'EulerAncestralDiscrete':
        scheduler = EulerAncestralDiscreteScheduler()
    elif args.sample_method == 'DEISMultistep':
        scheduler = DEISMultistepScheduler()
    elif args.sample_method == 'KDPM2AncestralDiscrete':  #########
        scheduler = KDPM2AncestralDiscreteScheduler()
    print('videogen_pipeline', device)
    # 定义视频生成管道，把这几个需要用到的模型合在一起
    videogen_pipeline = VideoGenPipeline(vae=vae,
                                         text_encoder=text_encoder,
                                         tokenizer=tokenizer,
                                         scheduler=scheduler,
                                         transformer=transformer_model).to(device=device)
    # videogen_pipeline.enable_xformers_memory_efficient_attention()
    
    # video_grids = []
    # 读取prompt
    if not isinstance(args.text_prompt, list):
        args.text_prompt = [args.text_prompt]
    if len(args.text_prompt) == 1 and args.text_prompt[0].endswith('txt'):
        text_prompt = open(args.text_prompt[0], 'r').readlines()
        args.text_prompt = [i.strip() for i in text_prompt]
    for prompt in args.text_prompt:
        print('Processing the ({}) prompt'.format(prompt))
        prompt_save = new_func(prompt)
        # 定义输出潜在变量（需要解码才能变成最后的视频帧）和最后生成结果的地方
        latents_dir = f"results/opensora_fifo/latents/{args.num_sampling_steps}steps/{prompt_save}"
        if args.version == "221x512x512":
            latents_dir = latents_dir.replace("opensora_fifo", "opensora_fifo_221")

        if args.output_dir is None:
            output_dir = f"results/opensora_fifo/video/{prompt_save}"

            if args.new_video_length != 100:
                output_dir = output_dir.replace(f"{prompt_save}", f"{prompt_save}/{args.new_video_length}frames")
            if not args.lookahead_denoising:
                output_dir = output_dir.replace(f"{prompt_save}", f"{prompt_save}/lookahead_denoising")
            if not args.num_partitions != 8:
                output_dir = output_dir.replace(f"{prompt_save}", f"{prompt_save}/{args.num_partitions}partitions")    

            if args.version == "221x512x512":
                output_dir = output_dir.replace("opensora_fifo", "opensora_fifo_221")
        else:
            output_dir = args.output_dir

        print("The results will be saved in", output_dir)
        print("The latents will be saved in", latents_dir)
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(latents_dir, exist_ok=True)
        # 判断latents目录里是否已经存在该视频的潜在表示
        is_run_base = not os.path.exists(os.path.join(latents_dir, "video.pt"))
        # 不存在则进入生成基础视频的流程
        if is_run_base:
            videos = videogen_pipeline(prompt,
                                    num_frames=args.num_frames,
                                    height=args.height,
                                    width=args.width,
                                    num_inference_steps=args.num_sampling_steps,
                                    guidance_scale=args.guidance_scale,
                                    enable_temporal_attentions=not args.force_images,
                                    num_images_per_prompt=1,
                                    mask_feature=True,
                                    save_latents=True,
                                    latents_dir=latents_dir,
                                    return_dict=False,
                                    )

            output_path = os.path.join(output_dir, "origin.mp4")
            imageio.mimwrite(output_path, videos[0][0], fps=args.fps, quality=9)  # highest quality is 10, lowest is 0
        # 设置时间步
        videogen_pipeline.scheduler.set_timesteps(args.num_sampling_steps, device=videogen_pipeline.text_encoder.device)
    
        latents = prepare_latents(args, latents_dir, scheduler=videogen_pipeline.scheduler)
        # 把帧保存起来
        if args.save_frames:
            fifo_dir = os.path.join(output_dir, "fifo")
            os.makedirs(fifo_dir, exist_ok=True)
        
        fifo_video_frames = []
        fifo_first_latents = []

        timesteps = videogen_pipeline.scheduler.timesteps
        # 翻转时间步，推理过程是从最后一帧（噪声最多的一帧）开始的
        timesteps = torch.flip(timesteps, [0])
        if args.lookahead_denoising:
            timesteps = torch.cat([torch.full((args.video_length//2,), timesteps[0]).to(timesteps.device), timesteps])


        num_iterations = args.new_video_length + args.queue_length - args.video_length if args.version == "65x512x512" else args.new_video_length + args.queue_length
        for i in trange(num_iterations):
            num_inference_steps_per_gpu = args.video_length

            for rank in reversed(range(2 * args.num_partitions if args.lookahead_denoising else args.num_partitions)):
                if args.lookahead_denoising:
                    start_idx = (rank // 2) * num_inference_steps_per_gpu + (rank % 2) * (num_inference_steps_per_gpu // 2)
                else:
                    start_idx = rank * num_inference_steps_per_gpu
                midpoint_idx = start_idx + num_inference_steps_per_gpu // 2 + (rank % 2)
                end_idx = start_idx + num_inference_steps_per_gpu

                t = timesteps[start_idx:end_idx]
                input_latents = latents[:,:,start_idx:end_idx].clone()

                output_latents, first_latent, first_frame = videogen_pipeline.fifo_onestep(prompt,
                                        video_length=args.video_length,
                                        height=args.height,
                                        width=args.width,
                                        num_inference_steps=args.num_sampling_steps,
                                        guidance_scale=args.guidance_scale,
                                        enable_temporal_attentions=not args.force_images,
                                        num_images_per_prompt=1,
                                        mask_feature=True,
                                        latents=input_latents,
                                        timesteps=t,
                                        save_frames=args.save_frames,
                                        )

                if args.lookahead_denoising:
                    latents[:,:,midpoint_idx:end_idx] = output_latents[:,:,-(end_idx-midpoint_idx):]
                else:
                    latents[:,:,start_idx:end_idx] = output_latents
                del output_latents

            latents = shift_latents(latents, videogen_pipeline.scheduler)
            
            if args.save_frames:
                output_path = os.path.join(fifo_dir, f"frame_{i:04d}.png")
                imageio.mimwrite(output_path, first_frame, quality=9)  # highest quality is 10, lowest is 0

            fifo_first_latents.append(first_latent)

        num_vae = args.new_video_length // (args.video_length-1)
        
        if args.version == "65x512x512":
            first_idx = args.queue_length - args.video_length
        else:
            first_idx = args.queue_length

        fifo_vae_video_frames = []
        for i in range(num_vae):
            target_latents = torch.cat(fifo_first_latents[first_idx+i*(args.video_length-1):first_idx+(i+1)*(args.video_length-1)+1], dim=2)
            video = videogen_pipeline.decode_latents(target_latents)[0]

            if i == 0:
                fifo_vae_video_frames.append(video)
            else:
                fifo_vae_video_frames.append(video[1:])
        
        if num_vae > 0:
            fifo_vae_video_frames = torch.cat(fifo_vae_video_frames, dim=0)
            if args.output_dir is None:
                output_vae_path = os.path.join(output_dir, "fifo_vae.mp4")
            else:
                output_vae_path = os.path.join(args.output_dir, f"{prompt_save}.mp4")
            imageio.mimwrite(output_vae_path, fifo_vae_video_frames, fps=args.fps, quality=9)

def new_func(prompt):
    prompt_save = prompt.replace(' ', '_')[:100]
    return prompt_save



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='LanguageBind/Open-Sora-Plan-v1.1.0')
    parser.add_argument("--version", type=str, default=None, choices=[None, '65x512x512', '221x512x512', '513x512x512'])
    parser.add_argument("--num_frames", type=int, default=65)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--ae", type=str, default='CausalVAEModel_4x8x8')
    parser.add_argument("--text_encoder_name", type=str, default='DeepFloyd/t5-v1_1-xxl')
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--sample_method", type=str, default="DDPM")
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--queue_length", type=int, default=17)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--run_time", type=int, default=0)
    parser.add_argument("--text_prompt", nargs='+')
    parser.add_argument('--force_images', action='store_true')
    parser.add_argument('--tile_overlap_factor', type=float, default=0.25)
    parser.add_argument('--enable_tiling', action='store_true')
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--video_length", "-f", type=int, default=17)
    parser.add_argument("--new_video_length", "-N", type=int, default=None)
    parser.add_argument("--num_partitions", "-n", type=int, default=4)
    parser.add_argument("--lookahead_denoising", "-ld", action='store_false', default=True)
    parser.add_argument("--output_dir", type=str, default=None, help="custom output directory")
    parser.add_argument("--save_frames", action='store_true', default=False)

    args = parser.parse_args()

    assert args.num_frames == 4*args.video_length - 3

    args.queue_length = args.video_length * args.num_partitions
    args.num_sampling_steps = args.video_length * args.num_partitions

    main(args)
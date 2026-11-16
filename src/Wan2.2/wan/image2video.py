# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import copy
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from .distributed.util import get_world_size
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae2_1 import Wan2_1_VAE
from .prompt_relay import (
    PromptRelayConfig,
    build_prompt_relay_segments,
    with_duplicate_frame_indices,
)
from .visual_relay import VisualRelayLayout
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class WanI2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.boundary = config.boundary
        self.param_dtype = config.param_dtype

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.low_noise_model = WanModel.from_pretrained(
            checkpoint_dir, subfolder=config.low_noise_checkpoint)
        self.low_noise_model = self._configure_model(
            model=self.low_noise_model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype)

        self.high_noise_model = WanModel.from_pretrained(
            checkpoint_dir, subfolder=config.high_noise_checkpoint)
        self.high_noise_model = self._configure_model(
            model=self.high_noise_model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype)
        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt

    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                         convert_model_dtype):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (torch.nn.Module):
                The model instance to configure.
            use_sp (`bool`):
                Enable distribution strategy of sequence parallel.
            dit_fsdp (`bool`):
                Enable FSDP sharding for DiT model.
            shard_fn (callable):
                The function to apply FSDP sharding.
            convert_model_dtype (`bool`):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.

        Returns:
            torch.nn.Module:
                The configured model.
        """
        model.eval().requires_grad_(False)

        if use_sp:
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(
                    sp_attn_forward, block.self_attn)
            model.forward = types.MethodType(sp_dit_forward, model)

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            model = shard_fn(model)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                model.to(self.device)

        return model

    def _prepare_model_for_timestep(self, t, boundary, offload_model):
        r"""
        Prepares and returns the required model for the current timestep.

        Args:
            t (torch.Tensor):
                current timestep.
            boundary (`int`):
                The timestep threshold. If `t` is at or above this value,
                the `high_noise_model` is considered as the required model.
            offload_model (`bool`):
                A flag intended to control the offloading behavior.

        Returns:
            torch.nn.Module:
                The active model on the target device for the current timestep.
        """
        if t.item() >= boundary:
            required_model_name = 'high_noise_model'
            offload_model_name = 'low_noise_model'
        else:
            required_model_name = 'low_noise_model'
            offload_model_name = 'high_noise_model'
        if offload_model or self.init_on_cpu:
            if next(getattr(
                    self,
                    offload_model_name).parameters()).device.type == 'cuda':
                getattr(self, offload_model_name).to('cpu')
            if next(getattr(
                    self,
                    required_model_name).parameters()).device.type == 'cpu':
                getattr(self, required_model_name).to(self.device)
        return getattr(self, required_model_name)

    def _prepare_prompt_relay(
        self,
        relay_config,
        *,
        latent_frames,
        lat_h,
        lat_w,
    ):
        """Tokenize and map local prompts to the I2V latent-frame grid."""

        tokens_per_frame = (
            (lat_h // self.patch_size[1])
            * (lat_w // self.patch_size[2])
        )
        return build_prompt_relay_segments(
            self.text_encoder.tokenizer,
            relay_config,
            latent_frames=latent_frames,
            tokens_per_frame=tokens_per_frame,
            text_len=self.config.text_len,
        )

    def generate(
        self,
        input_prompt,
        img,
        max_area=720 * 1280,
        frame_num=81,
        shift=5.0,
        sample_solver='unipc',
        sampling_steps=40,
        guide_scale=5.0,
        n_prompt="",
        seed=-1,
        offload_model=True,
        prompt_relay_config=None,
        prompt_filepath=None,
        query_chunk_size=128,
    ):
        """Generate standard I2V, Prompt Relay I2V, or multi-anchor Visual Relay."""

        if prompt_relay_config is not None and prompt_filepath is not None:
            raise ValueError("Pass prompt_relay_config or prompt_filepath, not both")
        if prompt_relay_config is None and prompt_filepath is not None:
            prompt_relay_config = PromptRelayConfig.from_json(prompt_filepath)
        if prompt_relay_config is not None and not isinstance(
            prompt_relay_config, PromptRelayConfig
        ):
            raise TypeError("prompt_relay_config must be a PromptRelayConfig")
        if type(query_chunk_size) is not int or query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be a positive integer")

        visual_relay = bool(
            prompt_relay_config is not None
            and prompt_relay_config.visual_relay_enabled
        )
        image_values = list(img) if isinstance(img, (list, tuple)) else [img]
        if visual_relay:
            if len(image_values) != len(prompt_relay_config.images):
                raise ValueError(
                    "loaded images do not match the images array in prompts.json"
                )
            image_indices = prompt_relay_config.image_indices
        else:
            image_values = image_values[:1]
            image_indices = (0,)

        output_latent_frames = (frame_num - 1) // self.vae_stride[0] + 1
        layout = (
            VisualRelayLayout.build(output_latent_frames, image_indices)
            if visual_relay
            else None
        )
        model_latent_frames = (
            layout.internal_latent_frames if layout else output_latent_frames
        )
        if visual_relay and sampling_steps <= 10:
            raise ValueError("Visual Relay requires more than 10 sampling steps")

        guide_scale = (
            (guide_scale, guide_scale)
            if isinstance(guide_scale, float)
            else guide_scale
        )
        images = [
            TF.to_tensor(value).sub_(0.5).div_(0.5).to(self.device)
            for value in image_values
        ]
        height, width = images[0].shape[1:]
        aspect_ratio = height / width
        lat_h = round(
            np.sqrt(max_area * aspect_ratio)
            // self.vae_stride[1]
            // self.patch_size[1]
            * self.patch_size[1]
        )
        lat_w = round(
            np.sqrt(max_area / aspect_ratio)
            // self.vae_stride[2]
            // self.patch_size[2]
            * self.patch_size[2]
        )
        height = lat_h * self.vae_stride[1]
        width = lat_w * self.vae_stride[2]
        tokens_per_frame = (
            lat_h // self.patch_size[1]
        ) * (lat_w // self.patch_size[2])
        max_seq_len = int(
            math.ceil(
                model_latent_frames * tokens_per_frame / self.sp_size
            )
        ) * self.sp_size

        relay_segments = None
        if prompt_relay_config is not None:
            relay_segments, input_prompt = self._prepare_prompt_relay(
                prompt_relay_config,
                latent_frames=output_latent_frames,
                lat_h=lat_h,
                lat_w=lat_w,
            )
        duplicate_indices = (
            layout.temporary_internal_indices if layout else ()
        )
        main_relay = with_duplicate_frame_indices(
            relay_segments or (), duplicate_indices
        )
        handoff_relay = tuple(relay_segments or ())

        def active_prompt(output_index):
            if prompt_relay_config is None:
                return input_prompt
            intervals = prompt_relay_config.frame_intervals(
                output_latent_frames
            )
            active = [
                prompt
                for prompt, (start, end) in zip(
                    prompt_relay_config.local_prompts, intervals
                )
                if start <= output_index < end
            ]
            return " ".join(
                [prompt_relay_config.global_prompt, *active]
            ).strip()

        text_prompts = [input_prompt]
        if visual_relay:
            text_prompts.extend(
                active_prompt(index)
                for index in prompt_relay_config.image_indices[1:]
            )
        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        null_prompts = [n_prompt] * len(text_prompts)
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            contexts = self.text_encoder(text_prompts, self.device)
            null_contexts = self.text_encoder(null_prompts, self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            contexts = self.text_encoder(text_prompts, torch.device('cpu'))
            null_contexts = self.text_encoder(
                null_prompts, torch.device('cpu')
            )
            contexts = [value.to(self.device) for value in contexts]
            null_contexts = [
                value.to(self.device) for value in null_contexts
            ]

        resized = [
            torch.nn.functional.interpolate(
                value[None].cpu(),
                size=(height, width),
                mode='bicubic',
            )[0].to(self.device)
            for value in images
        ]

        def first_frame_mask(latent_frames):
            mask = torch.zeros(
                4, latent_frames, lat_h, lat_w, device=self.device
            )
            mask[:, 0].fill_(1)
            return mask

        def standard_conditioning(image, latent_frames):
            pixel_frames = 1 + self.vae_stride[0] * (latent_frames - 1)
            video = torch.cat(
                [
                    image[:, None],
                    torch.zeros(
                        3,
                        pixel_frames - 1,
                        height,
                        width,
                        device=self.device,
                    ),
                ],
                dim=1,
            )
            encoded = self.vae.encode([video])[0]
            return torch.cat([first_frame_mask(latent_frames), encoded])

        handoff_y = standard_conditioning(
            resized[0], output_latent_frames
        )

        if visual_relay:
            image_by_output = {
                index: resized[position]
                for position, index in enumerate(
                    prompt_relay_config.image_indices
                )
            }
            chunks = [resized[0][:, None]]
            for output_index in range(1, output_latent_frames):
                if output_index in image_by_output:
                    chunks.append(image_by_output[output_index][:, None])
                chunks.append(
                    torch.zeros(
                        3,
                        self.vae_stride[0],
                        height,
                        width,
                        device=self.device,
                    )
                )
            joint_video = torch.cat(chunks, dim=1)
            joint_encoded = self.vae.encode(
                [joint_video],
                single_frame_indices=layout.temporary_internal_indices,
                single_frame_four_history=True,
            )[0]
            if joint_encoded.size(1) != model_latent_frames:
                raise RuntimeError(
                    "multi-anchor VAE encoding produced the wrong length"
                )
            joint_mask = first_frame_mask(model_latent_frames)
            for anchor in layout.temporary_internal_indices:
                joint_mask[:, anchor].fill_(1)
            joint_y = torch.cat([joint_mask, joint_encoded])
            auxiliary_y = []
            for position, output_index in enumerate(
                prompt_relay_config.image_indices[1:], start=1
            ):
                suffix_frames = output_latent_frames - output_index + 2
                auxiliary_y.append(
                    standard_conditioning(
                        resized[position], suffix_frames
                    )
                )
        else:
            joint_y = handoff_y
            auxiliary_y = []

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        latent = torch.randn(
            16,
            model_latent_frames,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device,
        )
        b0_latents = [
            torch.randn(
                16,
                1,
                lat_h,
                lat_w,
                dtype=torch.float32,
                generator=seed_g,
                device=self.device,
            )
            for _ in range(layout.temporary_count if layout else 0)
        ]

        def make_scheduler():
            if sample_solver == 'unipc':
                scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False,
                )
                scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift
                )
                return scheduler
            if sample_solver == 'dpm++':
                scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False,
                )
                sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    scheduler, device=self.device, sigmas=sigmas
                )
                scheduler.timesteps = timesteps
                return scheduler
            raise NotImplementedError("Unsupported solver")

        sample_scheduler = make_scheduler()
        timesteps = sample_scheduler.timesteps
        b0_schedulers = [make_scheduler() for _ in b0_latents]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_low = getattr(
            self.low_noise_model, 'no_sync', noop_no_sync
        )
        no_sync_high = getattr(
            self.high_noise_model, 'no_sync', noop_no_sync
        )
        boundary = self.boundary * self.num_train_timesteps
        retained_indices = (
            layout.retained_internal_indices
            if layout
            else tuple(range(output_latent_frames))
        )

        with (
            torch.amp.autocast('cuda', dtype=self.param_dtype),
            torch.no_grad(),
            no_sync_low(),
            no_sync_high(),
        ):
            for sampling_step, timestep_value in enumerate(tqdm(timesteps)):
                model = self._prepare_model_for_timestep(
                    timestep_value, boundary, offload_model
                )
                scale = (
                    guide_scale[1]
                    if timestep_value.item() >= boundary
                    else guide_scale[0]
                )
                guidance_active = visual_relay and sampling_step < 10
                if guidance_active:
                    main_input = latent
                    current_y = joint_y
                    current_relay = main_relay
                else:
                    main_input = latent[:, retained_indices]
                    current_y = handoff_y
                    current_relay = handoff_relay

                timestep = torch.stack([timestep_value]).to(self.device)
                common = {
                    'seq_len': max_seq_len,
                    'y': [current_y],
                }
                conditional = model(
                    [main_input],
                    t=timestep,
                    context=[contexts[0]],
                    prompt_relay_segments=current_relay,
                    prompt_relay_query_chunk_size=query_chunk_size,
                    **common,
                )[0]
                unconditional = model(
                    [main_input],
                    t=timestep,
                    context=[null_contexts[0]],
                    **common,
                )[0]

                b0_predictions_cond = []
                b0_predictions_uncond = []
                if guidance_active:
                    for anchor_number, anchor_internal in enumerate(
                        layout.temporary_internal_indices
                    ):
                        selected = layout.auxiliary_internal_indices(
                            anchor_number
                        )
                        own_position = selected.index(anchor_internal)
                        suffix_indices = selected[own_position:]
                        b_input = torch.cat(
                            [
                                b0_latents[anchor_number],
                                latent[:, suffix_indices],
                            ],
                            dim=1,
                        )
                        b_timestep = torch.stack(
                            [timestep_value]
                        ).to(self.device)
                        b_common = {
                            'seq_len': max_seq_len,
                            'y': [auxiliary_y[anchor_number]],
                        }
                        b_cond = model(
                            [b_input],
                            t=b_timestep,
                            context=[contexts[anchor_number + 1]],
                            **b_common,
                        )[0]
                        b_uncond = model(
                            [b_input],
                            t=b_timestep,
                            context=[null_contexts[anchor_number + 1]],
                            **b_common,
                        )[0]
                        conditional[:, anchor_internal:anchor_internal + 1] = (
                            b_cond[:, 1:2]
                        )
                        unconditional[:, anchor_internal:anchor_internal + 1] = (
                            b_uncond[:, 1:2]
                        )
                        b0_predictions_cond.append(b_cond[:, :1])
                        b0_predictions_uncond.append(b_uncond[:, :1])
                        del b_cond, b_uncond, b_input

                prediction = unconditional + scale * (
                    conditional - unconditional
                )
                if visual_relay and not guidance_active:
                    expanded = torch.empty_like(latent)
                    expanded[:, retained_indices] = prediction
                    for anchor_internal in layout.temporary_internal_indices:
                        left = max(anchor_internal - 1, 0)
                        right = min(anchor_internal + 1, latent.size(1) - 1)
                        expanded[:, anchor_internal:anchor_internal + 1] = (
                            0.5
                            * (
                                latent[:, left:left + 1]
                                + latent[:, right:right + 1]
                            )
                        )
                    prediction = expanded

                next_latent = sample_scheduler.step(
                    prediction.unsqueeze(0),
                    timestep_value,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g,
                )[0].squeeze(0)
                del conditional, unconditional, prediction
                latent = next_latent

                if guidance_active:
                    for anchor_number, scheduler in enumerate(b0_schedulers):
                        b0_prediction = (
                            b0_predictions_uncond[anchor_number]
                            + scale
                            * (
                                b0_predictions_cond[anchor_number]
                                - b0_predictions_uncond[anchor_number]
                            )
                        )
                        b0_latents[anchor_number] = scheduler.step(
                            b0_prediction.unsqueeze(0),
                            timestep_value,
                            b0_latents[anchor_number].unsqueeze(0),
                            return_dict=False,
                            generator=seed_g,
                        )[0].squeeze(0)
                if visual_relay and sampling_step == 10:
                    logging.info(
                        "Visual Relay handoff removed %d temporary anchors",
                        layout.temporary_count,
                    )

            if offload_model:
                self.low_noise_model.cpu()
                self.high_noise_model.cpu()
                torch.cuda.empty_cache()

            retained = latent[:, retained_indices]
            if retained.size(1) != output_latent_frames:
                raise RuntimeError("Visual Relay retained the wrong output length")
            videos = self.vae.decode([retained]) if self.rank == 0 else None

        del latent, sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
        return videos[0] if self.rank == 0 else None

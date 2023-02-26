# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import os
import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram

from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from wavegrad.dataset import from_path as dataset_from_path
from wavegrad.model import WaveGrad


def _nested_map(struct, map_fn):
    if isinstance(struct, tuple):
        return tuple(_nested_map(x, map_fn) for x in struct)
    if isinstance(struct, list):
        return [_nested_map(x, map_fn) for x in struct]
    if isinstance(struct, dict):
        return {k: _nested_map(v, map_fn) for k, v in struct.items()}
    return map_fn(struct)


class WaveGradLearner:
    def __init__(self, model_dir, model, dataset, optimizer, scheduler, params, *args, **kwargs):
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.params = params
        self.autocast = torch.cuda.amp.autocast(enabled=kwargs.get('fp16', False))
        self.scaler = torch.cuda.amp.GradScaler(enabled=kwargs.get('fp16', False))
        self.step = 0
        self.is_master = True

        beta = np.array(self.params.noise_schedule)
        noise_level = np.cumprod(1 - beta) ** 0.5
        noise_level = np.concatenate([[1.0], noise_level], axis=0)
        self.noise_level = torch.tensor(noise_level.astype(np.float32))
        self.loss_fn = self.spectral_reconstruction_loss
        self.summary_writer = None

    def state_dict(self):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        return {
            'step': self.step,
            'model': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items()},
            'optimizer': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in
                          self.optimizer.state_dict().items()},
            'params': dict(self.params),
            'scaler': self.scaler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            self.model.module.load_state_dict(state_dict['model'])
        else:
            self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scaler.load_state_dict(state_dict['scaler'])
        self.step = state_dict['step']

    def save_to_checkpoint(self, filename='weights'):
        save_basename = f'{filename}-{self.step}.pt'
        save_name = f'{self.model_dir}/{save_basename}'
        link_name = f'{self.model_dir}/{filename}.pt'
        torch.save(self.state_dict(), save_name)
        if os.name == 'nt':
            torch.save(self.state_dict(), link_name)
        else:
            if os.path.islink(link_name):
                os.unlink(link_name)
            os.symlink(save_basename, link_name)

    def restore_from_checkpoint(self, filename='weights'):
        try:
            checkpoint = torch.load(f'{self.model_dir}/{filename}.pt')
            self.load_state_dict(checkpoint)
            return True
        except FileNotFoundError:
            return False

    def train(self, max_steps=None):
        device = next(self.model.parameters()).device
        while True:
            for features in tqdm(self.dataset,
                                 desc=f'Epoch {self.step // len(self.dataset)}') if self.is_master else self.dataset:
                if max_steps is not None and self.step >= max_steps:
                    return
                features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
                loss = self.train_step(features)
                if self.is_master:
                    if self.step % 100 == 0:
                        self._write_summary(self.step, features, loss)
                    if self.step % len(self.dataset) == 0:
                        self.save_to_checkpoint()
                self.step += 1

    def train_step(self, features):
        for param in self.model.parameters():
            param.grad = None

        audio = features['audio']
        spectrogram = features['spectrogram']

        N, T = audio.shape
        S = 1000
        device = audio.device
        self.noise_level = self.noise_level.to(device)

        with self.autocast:
            s = torch.randint(1, S + 1, [N], device=audio.device)
            l_a, l_b = self.noise_level[s - 1], self.noise_level[s]
            noise_scale = l_a + torch.rand(N, device=audio.device) * (l_b - l_a)
            noise_scale = noise_scale.unsqueeze(1)

            noise = torch.randn_like(audio)
            noise_coef = (1.0 - noise_scale ** 2) ** 0.5
            noisy_audio = noise_scale * audio + noise_coef * noise
            predicted_noise = self.model(noisy_audio, spectrogram, noise_scale.squeeze(1))
            noise_scale += 1e-4
            predicted_audio = (noisy_audio - noise_coef * predicted_noise) / noise_scale
            loss = self.loss_fn(audio, predicted_audio.squeeze(1))

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_grad_norm,
                                                  error_if_nonfinite=True)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        # self.scheduler.step()
        return loss

    def _write_summary(self, step, features, loss):
        writer = self.summary_writer or SummaryWriter(self.model_dir, purge_step=step)
        writer.add_audio('audio/reference', features['audio'][0], step, sample_rate=self.params.sample_rate)
        writer.add_scalar('train/loss', loss, step)
        writer.add_scalar('train/grad_norm', self.grad_norm, step)
        writer.flush()
        self.summary_writer = writer

    def spectral_reconstruction_loss(self, reference, predicted):
        device = next(self.model.parameters()).device
        L = 0
        eps = 1e-4
        for i in range(6, 12):
            s = 2 ** i
            alpha_s = (s / 2) ** 0.5
            hop = s // 4
            melspec = MelSpectrogram(sample_rate=self.params.sample_rate, n_fft=s, hop_length=hop, n_mels=64,
                                     wkwargs={"device": device}).to(device)
            S_x = melspec(reference)
            S_G_x = melspec(predicted)

            loss = (S_x - S_G_x).abs().sum() + alpha_s * (
                    ((torch.log(S_x.abs() + eps) - torch.log(S_G_x.abs() + eps)) ** 2).sum(dim=-2) ** 0.5).sum()
            L += loss
        return L


def _train_impl(replica_id, model, dataset, args, params, is_distributed):
  torch.backends.cudnn.benchmark = True
  opt = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
  sched = torch.optim.lr_scheduler.StepLR(opt, step_size=params.sched_step, gamma=params.sched_gamma)

  learner = WaveGradLearner(args.model_dir, model, dataset, opt, sched, params, fp16=args.fp16)
  learner.is_master = (replica_id == 0) if is_distributed else True
  learner.restore_from_checkpoint()
  learner.train(max_steps=args.max_steps)


def train(args, params):
  dataset = dataset_from_path(args.data_dirs, params)
  model = WaveGrad(params).cuda()
  _train_impl(os.environ.get('CUDA_VISIBLE_DEVICES', 0), model, dataset, args, params, is_distributed=False)


def train_distributed(replica_id, replica_count, port, args, params):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    torch.distributed.init_process_group('nccl', rank=replica_id, world_size=replica_count)

    device = torch.device('cuda', replica_id)
    torch.cuda.set_device(device)
    model = WaveGrad(params).to(device)
    model = DistributedDataParallel(model, device_ids=[replica_id])
    _train_impl(replica_id, model, dataset_from_path(args.data_dirs, params, is_distributed=True), args, params, is_distributed=True)

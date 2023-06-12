# Copyright 2022 Adobe. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

import unittest
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import librosa
import numpy as np
from scipy import signal as sig
import tempfile
import torchaudio

from convmelspec.stft import ConvertibleSpectrogram as Spectrogram
from convmelspec.mil import *

BATCH_SIZE = 1
MEL_BANDS = 128
LABEL_DIM = 527
FFT_SIZE = 1024
HOP_SIZE = 512
EPS = 1e-6
SR = 16000
DEVICE = "cpu"


def get_hann_torch(win_size, sym=True):
    wn = sig.windows.hann(win_size, sym=sym).astype(np.float32)
    return torch.from_numpy(wn)


class TestEquivalence(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.OUTPUT_DIR = cls.temp_dir.name

        example_audio_path = librosa.example("nutcracker")
        y, sr = librosa.load(example_audio_path, sr=SR)
        total_sec = 1
        y = y[int(sr) : (total_sec * sr + sr)].astype(np.float32)
        cls.audio = y

        # Create output test dir if it doesn't exist
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        return

    def test_melpec_vs_librosa(self):

        wn = sig.windows.hann(FFT_SIZE, sym=True)

        # Create PyTorch STFT layer
        stft = Spectrogram(
            sr=SR,
            n_fft=FFT_SIZE,
            hop_size=HOP_SIZE,
            n_mel=None,
            window=wn,
            padding=0,
            spec_mode="DFT",
        )
        melstft = Spectrogram(
            sr=SR,
            n_fft=FFT_SIZE,
            hop_size=HOP_SIZE,
            n_mel=MEL_BANDS,
            window=wn,
            padding=0,
            spec_mode="DFT",
            mel_mode="librosa",
            mel_scale="slaney",
            dtype=torch.float32,
            debug=True,
        )
        stft.to(DEVICE)
        melstft.to(DEVICE)
        # Store CPU audio to GPU tensor of batch x channels x samples
        x = torch.zeros([BATCH_SIZE, self.audio.shape[0]])
        x[0, :] = torch.from_numpy(self.audio[: x.shape[1]])
        x = x.to(DEVICE)

        # Create Librosa power spec + mel spec from individual parts
        S_librosa = librosa.stft(
            self.audio,
            n_fft=FFT_SIZE,
            hop_length=HOP_SIZE,
            win_length=FFT_SIZE,
            window=wn,
            center=False,
        )
        S_librosa = np.abs(S_librosa)
        mel_matrix = librosa.filters.mel(
            sr=SR, n_fft=FFT_SIZE, n_mels=MEL_BANDS
        )
        S2_librosa = np.abs(S_librosa) ** 2
        M_librosa = mel_matrix.dot(S2_librosa)

        S2_gpu = stft(x)
        S_gpu = stft(x, power=False)
        M_gpu = melstft(x)

        S_gpu = S_gpu.detach().cpu().numpy()[0, :, :]
        S2_gpu = S2_gpu.detach().cpu().numpy()[0, :, :]
        M_gpu = M_gpu.detach().cpu().numpy()[0, :, :]

        np.testing.assert_allclose(S_librosa, S_gpu, atol=1e-04)
        np.testing.assert_allclose(S2_librosa, S2_gpu, atol=1e-04)
        np.testing.assert_allclose(M_librosa, M_gpu, atol=1e-04)

    def test_melpec_vs_torchaudio(self):

        wn = sig.windows.hann(FFT_SIZE, sym=True)

        # Create PyTorch STFT layer
        stft = Spectrogram(
            sr=SR,
            n_fft=FFT_SIZE,
            hop_size=HOP_SIZE,
            n_mel=None,
            padding=0,
            window=wn,
            spec_mode="DFT",
            dtype=torch.float32,
        )
        melstft = Spectrogram(
            sr=SR,
            n_fft=FFT_SIZE,
            hop_size=HOP_SIZE,
            n_mel=MEL_BANDS,
            padding=0,
            window=wn,
            spec_mode="DFT",
            mel_mode="torchaudio",
            mel_scale="htk",
            dtype=torch.float32,
            debug=True,
        )

        stft.to(DEVICE)
        melstft.to(DEVICE)

        # print(stft.state_dict())
        # print(stft.params)

        # Store CPU audio to GPU tensor of batch x channels x samples
        x = torch.zeros([BATCH_SIZE, self.audio.shape[0]])
        x[0, :] = torch.from_numpy(self.audio[: x.shape[1]])
        x = x.to(DEVICE)

        window_fn = get_hann_torch

        # Create torchaudio power spec + mel spec from individual parts
        spec_layer = torchaudio.transforms.Spectrogram(
            n_fft=FFT_SIZE,
            hop_length=HOP_SIZE,
            window_fn=window_fn,
            power=2.0,
            center=False,
        )
        spec_layer.to(DEVICE)
        melspec_layer = torchaudio.transforms.MelSpectrogram(
            n_fft=FFT_SIZE,
            hop_length=HOP_SIZE,
            window_fn=window_fn,
            power=2.0,
            center=False,
            n_mels=MEL_BANDS,
        )
        melspec_layer.to(DEVICE)

        S2_librosa = spec_layer(x[0, :]).to("cpu")

        M_librosa = melspec_layer(x[0, :]).to("cpu")

        S2_gpu = stft(x, power=True).to("cpu")
        M_gpu = melstft(x, power=True).to("cpu")

        S2_gpu = S2_gpu.detach().cpu().numpy()[0, :, :]
        M_gpu = M_gpu.detach().cpu().numpy()[0, :, :]

        np.testing.assert_allclose(S2_librosa, S2_gpu, atol=1e-04)
        np.testing.assert_allclose(M_librosa, M_gpu, atol=1e-04)




if __name__ == "__main__":
    unittest.main()
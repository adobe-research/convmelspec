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

from convmelspec.convert import *
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


class TestConvertONNX(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # return

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

    def test_convert_stft_to_traced_on_the_fly(self):

        pth32_output_path = os.path.join(self.OUTPUT_DIR, "PTH-1-stft.pth")
        pth16_output_path = os.path.join(
            self.OUTPUT_DIR, "PTH-1-stft16bit.pth"
        )

        dft_mode = "on_the_fly"

        # Create PyTorch STFT layer
        model = Spectrogram(
            sr=SR,
            n_fft=FFT_SIZE,
            hop_size=HOP_SIZE,
            n_mel=None,
        )
        model.set_mode("DFT", dft_mode=dft_mode)
        model.eval()
        model.to(DEVICE)

        x = torch.zeros([BATCH_SIZE, self.audio.shape[0]])
        x[0, :] = torch.from_numpy(self.audio[: x.shape[1]])
        x = x.to(DEVICE)

        traced_model_f = torch.jit.trace(model, x)
        torch.jit.save(traced_model_f, pth32_output_path)

        with torch.cuda.amp.autocast():
            traced_model_f = torch.jit.trace(model, x)
            torch.jit.save(traced_model_f, pth16_output_path)

    def test_convert_stft_to_traced_precompute(self):

        verbose = False
        pth32_output_path = os.path.join(self.OUTPUT_DIR, "PTH-2-stft.pth")
        pth16_output_path = os.path.join(
            self.OUTPUT_DIR, "PTH-2-stft16bit.pth"
        )

        dft_mode = "store"

        # Create PyTorch STFT layer
        model = Spectrogram(
            sr=SR,
            n_fft=FFT_SIZE,
            hop_size=HOP_SIZE,
            n_mel=MEL_BANDS,
        )
        model.set_mode("DFT", dft_mode=dft_mode)
        model.eval()
        model.to(DEVICE)

        x = torch.zeros([BATCH_SIZE, self.audio.shape[0]])
        x[0, :] = torch.from_numpy(self.audio[: x.shape[1]])
        x = x.to(DEVICE)

        traced_model = torch.jit.trace(model, x)
        torch.jit.save(traced_model, pth32_output_path)

        with torch.cuda.amp.autocast():
            traced_model = torch.jit.trace(model, x)
            torch.jit.save(traced_model, pth16_output_path)

    def test_convert_stft_to_traced_input(self):

        pth32_output_path = os.path.join(self.OUTPUT_DIR, "PTH-3-stft.pth")
        pth16_output_path = os.path.join(
            self.OUTPUT_DIR, "PTH-3-stft16bit.pth"
        )

        dft_mode = "input"

        # Create PyTorch STFT layer
        model = Spectrogram(
            sr=SR,
            n_fft=FFT_SIZE,
            hop_size=HOP_SIZE,
            n_mel=MEL_BANDS,
            spec_mode="DFT",
        )
        model.eval()
        model.to(DEVICE)
        DFTr, DFTi = model.set_mode("DFT", dft_mode)

        x = torch.zeros([BATCH_SIZE, self.audio.shape[0]])
        x[0, :] = torch.from_numpy(self.audio[: x.shape[1]])
        x = x.to(DEVICE)
        DFTr = DFTr.to(DEVICE)
        DFTi = DFTi.to(DEVICE)

        traced_model = torch.jit.trace(model, (x, DFTr, DFTi))
        torch.jit.save(traced_model, pth32_output_path)

        with torch.cuda.amp.autocast():
            traced_model = torch.jit.trace(model, (x, DFTr, DFTi))
            torch.jit.save(traced_model, pth16_output_path)





if __name__ == "__main__":
    unittest.main()
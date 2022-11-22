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

    def test_convert_stft_to_onnx_on_the_fly(self):

        onnx32_output_path = os.path.join(self.OUTPUT_DIR, "ONNX-1-stft.onnx")
        onnx16_output_path = os.path.join(
            self.OUTPUT_DIR, "ONNX-1-stft16bit.onnx"
        )

        dft_mode = "on_the_fly"

        # Create PyTorch STFT layer
        stft = Spectrogram(
            sr=SR,
            n_fft=FFT_SIZE,
            hop_size=HOP_SIZE,
            n_mel=MEL_BANDS,
            spec_mode="DFT",
            dft_mode="store",
        )
        stft.to(DEVICE)
        # Train stuff

        # Switch to eval for conversion
        stft.eval()

        _, _ = stft.set_mode("DFT", dft_mode=dft_mode)

        x = torch.zeros([BATCH_SIZE, self.audio.shape[0]])
        x[0, :] = torch.from_numpy(self.audio[: x.shape[1]])
        x = x.to(DEVICE)

        save_onnx(stft, x, onnx32_output_path, False)
        # WARNING: 16-bit DFT might cause issues, warning
        with torch.cuda.amp.autocast():
            save_onnx(stft, x, onnx16_output_path, False)

    def test_convert_stft_to_onnx_precompute(self):

        onnx32_output_path = os.path.join(self.OUTPUT_DIR, "ONNX-2-stft.onnx")
        onnx16_output_path = os.path.join(
            self.OUTPUT_DIR, "ONNX-2-stft16bit.onnx"
        )

        # dft_mode = 'on_the_fly', 'store', 'input_DFT'
        dft_mode = "store"

        # Create PyTorch STFT layer
        stft = Spectrogram(
            sr=SR,
            n_fft=FFT_SIZE,
            hop_size=HOP_SIZE,
            n_mel=MEL_BANDS,
            spec_mode="DFT",
        )
        stft.set_mode("DFT", dft_mode=dft_mode)
        stft.eval()
        stft.to(DEVICE)

        x = torch.zeros([BATCH_SIZE, self.audio.shape[0]])
        x[0, :] = torch.from_numpy(self.audio[: x.shape[1]])
        x = x.to(DEVICE)

        save_onnx(stft, x, onnx32_output_path, False)
        with torch.cuda.amp.autocast():
            save_onnx(stft, x, onnx16_output_path, False)

    def test_convert_stft_to_onnx_input(self):

        verbose = False
        onnx32_output_path = os.path.join(self.OUTPUT_DIR, "ONNX-3-stft.onnx")
        onnx16_output_path = os.path.join(
            self.OUTPUT_DIR, "ONNX-3-stft16bit.onnx"
        )

        # dft_mode = 'on_the_fly', 'precompute', 'input'
        dft_mode = "input"

        # Create PyTorch STFT layer
        stft = Spectrogram(
            sr=SR,
            n_fft=FFT_SIZE,
            hop_size=HOP_SIZE,
            n_mel=MEL_BANDS,
            spec_mode="DFT",
        )
        stft.to(DEVICE)
        DFTr, DFTi = stft.set_mode("DFT", dft_mode=dft_mode)
        stft.eval()
        stft.to(DEVICE)

        x = torch.zeros([BATCH_SIZE, self.audio.shape[0]])
        x[0, :] = torch.from_numpy(self.audio[: x.shape[1]])
        x = x.to(DEVICE)

        input_tensors = (x, {"DFTr": DFTr, "DFTi": DFTi})

        input_names = ["input", "DFTr", "DFTi"]

        save_onnx(
            stft,
            input_tensors,
            onnx32_output_path,
            False,
            input_names=input_names,
        )

        with torch.cuda.amp.autocast():
            input_tensors = (x, {"DFTr": DFTr, "DFTi": DFTi})
            save_onnx(
                stft,
                input_tensors,
                onnx16_output_path,
                False,
                input_names=input_names,
            )





if __name__ == "__main__":
    unittest.main()
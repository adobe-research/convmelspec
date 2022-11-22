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
import coremltools as ct
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


class TestConvertCoreML(unittest.TestCase):

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

    def test_convert_stft_to_coreml_on_the_fly(self):

        coreml32_output_path = os.path.join(
            self.OUTPUT_DIR, "CML-1-stft.mlmodel"
        )
        coreml16_output_path = os.path.join(
            self.OUTPUT_DIR, "CML-1-stft16bit.mlmodel"
        )

        dft_mode = "on_the_fly"

        # Create PyTorch STFT layer in torchaudio/training mode
        model = Spectrogram(
            sr=SR,
            n_fft=FFT_SIZE,
            hop_size=HOP_SIZE,
            n_mel=MEL_BANDS,
        )
        model.to(DEVICE)
        # DO TRAINING WITH STORED DFT MATRIX

        # Switch to eval to convert
        model.eval()

        # Switch the spec model to DFT and set DFT mode to be on-the-fly + coreml mode
        model.set_mode("DFT", dft_mode=dft_mode, coreml=True)

        x = torch.zeros([BATCH_SIZE, self.audio.shape[0]])
        x[0, :] = torch.from_numpy(self.audio[: x.shape[1]])
        x = x.to(DEVICE)

        traced_model = torch.jit.trace(model, x)
        dummy_input = x
        dummy_output = model(x)

        input_tensors = [ct.TensorType(name="input", shape=(x.shape))]

        save_coreml(
            traced_model,
            input_tensors,
            dummy_output.shape,
            32,
            coreml32_output_path,
        )
        save_coreml(
            traced_model,
            input_tensors,
            dummy_output.shape,
            16,
            coreml16_output_path,
        )

    def test_convert_stft_to_coreml_store(self):

        coreml32_output_path = os.path.join(
            self.OUTPUT_DIR, "CML-2-stft.mlmodel"
        )
        coreml16_output_path = os.path.join(
            self.OUTPUT_DIR, "CML-2-stft16bit.mlmodel"
        )

        dft_mode = "store"

        # Create PyTorch STFT layer
        model = Spectrogram(
            sr=SR,
            n_fft=FFT_SIZE,
            hop_size=HOP_SIZE,
            n_mel=MEL_BANDS,
        )
        model.eval()
        model.to(DEVICE)
        model.set_mode("DFT", dft_mode=dft_mode)

        x = torch.zeros([BATCH_SIZE, self.audio.shape[0]])
        x[0, :] = torch.from_numpy(self.audio[: x.shape[1]])
        x = x.to(DEVICE)

        traced_model = torch.jit.trace(model, x)
        dummy_output = model(x)

        input_tensors = [ct.TensorType(name="input", shape=(x.shape))]

        save_coreml(
            traced_model,
            input_tensors,
            dummy_output.shape,
            32,
            coreml32_output_path,
        )
        save_coreml(
            traced_model,
            input_tensors,
            dummy_output.shape,
            16,
            coreml16_output_path,
        )

    def test_convert_stft_to_coreml_input(self):

        coreml32_output_path = os.path.join(
            self.OUTPUT_DIR, "CML-3-stft.mlmodel"
        )
        coreml16_output_path = os.path.join(
            self.OUTPUT_DIR, "CML-3-stft16bit.mlmodel"
        )

        dft_mode = "input"

        # Create PyTorch STFT layer
        model = Spectrogram(
            sr=SR,
            n_fft=FFT_SIZE,
            hop_size=HOP_SIZE,
            n_mel=MEL_BANDS,
        )
        model.eval()
        model.to(DEVICE)
        DFTr, DFTi = model.set_mode("DFT", dft_mode=dft_mode)

        x = torch.zeros([BATCH_SIZE, self.audio.shape[0]])
        x[0, :] = torch.from_numpy(self.audio[: x.shape[1]])
        x = x.to(DEVICE)
        DFTr = DFTr.to(DEVICE)
        DFTi = DFTi.to(DEVICE)
        output = model(x, DFTr=DFTr, DFTi=DFTi)

        traced_model = torch.jit.trace(model, (x, DFTr, DFTi))
        input_tensors = [
            ct.TensorType(name="input", shape=(x.shape)),
            ct.TensorType(name="DFTr", shape=(DFTr.shape)),
            ct.TensorType(name="DFTi", shape=(DFTi.shape)),
        ]

        save_coreml(
            traced_model, input_tensors, output.shape, 32, coreml32_output_path
        )
        save_coreml(
            traced_model, input_tensors, output.shape, 16, coreml16_output_path
        )




if __name__ == "__main__":
    unittest.main()
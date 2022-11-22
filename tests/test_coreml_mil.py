# Copyright 2022 Adobe. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

# from statistics import mode
import unittest
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import librosa
import numpy as np
from scipy import signal as sig
import coremltools as ct
import tempfile
import torchaudio

import convmelspec.convert as fc


from convmelspec.stft import ConvertibleSpectrogram as Spectrogram
from convmelspec.mil import *


BATCH_SIZE = 1
MEL_BANDS = 128
LABEL_DIM = 527
FFT_SIZE = 1024
HOP_SIZE = 512
EPS = 1e-6
SR = 16000


# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"


def get_hann_torch(win_size, sym=True):
    wn = sig.windows.hann(win_size, sym=sym).astype(np.float32)
    return torch.from_numpy(wn)

def get_box_torch(win_size, sym=True):
    wn = 1+0*sig.windows.hann(win_size, sym=sym).astype(np.float32)
    return torch.from_numpy(wn)


class TestCoreMLMILConversion(unittest.TestCase):
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

    def test_torchaudio_spectrogram_coreml_mil_convert(self):

        for mode in ['on_the_fly', 'store']:

            #
            register_stft(mode)

            output_path = os.path.join(
                self.OUTPUT_DIR, "coreml-mil-stft-" + mode + ".mlmodel"
            )
            # Store CPU audio to GPU tensor of batch x channels x samples
            x = torch.zeros([BATCH_SIZE, self.audio.shape[0]])
            x[0, :] = torch.from_numpy(self.audio[: x.shape[1]])

            window_fn = torch.hann_window

            # Create torchaudio power spec + mel spec from individual parts
            spec_layer = torchaudio.transforms.Spectrogram(
                n_fft=FFT_SIZE,
                hop_length=HOP_SIZE,
                window_fn=window_fn,
                power=2.0,
            )

            traced_model = torch.jit.trace(spec_layer, x)

            traced_output = spec_layer(x).numpy()

            input_tensors = [ct.TensorType(name="input", shape=(x.shape))]

            fc.save_coreml(
                traced_model,
                input_tensors,
                traced_output.shape,
                32,
                output_path,
            )

            cm_model = ct.models.MLModel(output_path)

            input0 = x.numpy()
            output_dict = cm_model.predict(
                {
                    "input": input0,
                }
            )
            cml_output = output_dict['output']
            self.assertTrue(np.allclose(cml_output, traced_output, atol=1e-04))


    def test_torchaudio_melspectrogram_coreml_mil_convert(self):

        # register_stft('on_the_fly')
        for mode in ['on_the_fly']:

            register_stft(mode)

            output_path = os.path.join(
                self.OUTPUT_DIR, "coreml-mil-melspec-" + mode + ".mlmodel"
            )
            # Store CPU audio to GPU tensor of batch x channels x samples
            x = torch.zeros([BATCH_SIZE, self.audio.shape[0]])
            x[0, :] = torch.from_numpy(self.audio[: x.shape[1]])

            window_fn = torch.hann_window

            # Create torchaudio power spec + mel spec from individual parts
            spec_layer = torchaudio.transforms.MelSpectrogram(
                sample_rate=SR,
                n_fft=FFT_SIZE,
                hop_length=HOP_SIZE,
                window_fn=window_fn,
                power=2.0,
            )

            traced_model = torch.jit.trace(spec_layer, x)

            traced_output = spec_layer(x).numpy()

            input_tensors = [ct.TensorType(name="input", shape=(x.shape))]

            fc.save_coreml(
                traced_model,
                input_tensors,
                traced_output.shape,
                32,
                output_path,
            )

            cm_model = ct.models.MLModel(output_path)

            input0 = x.numpy()
            output_dict = cm_model.predict(
                {
                    "input": input0,
                }
            )
            cml_output = output_dict['output']
            self.assertTrue(np.allclose(cml_output, traced_output, atol=1e-04))


if __name__ == "__main__":
    unittest.main()
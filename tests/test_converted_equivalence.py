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
import itertools
import unittest
import os
from typing import Literal

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch, torchaudio
import librosa
import numpy as np
from scipy import signal as sig
import coremltools as ct
import onnxruntime as ort
import platform
import tempfile

import convmelspec.convert as fc


from convmelspec.stft import ConvertibleSpectrogram as Spectrogram


BATCH_SIZE = 1
MEL_BANDS = 128
LABEL_DIM = 527
FFT_SIZE = 1024
HOP_SIZE = 512
EPS = 1e-6
SR = 16000


DEVICE = "cpu"


class TestConvertedEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.OUTPUT_DIR = cls.temp_dir.name

        example_audio_path = librosa.example("nutcracker")
        y, sr = librosa.load(example_audio_path, sr=SR)
        # total_sec = 1
        # y = y[int(sr) : (total_sec * sr + sr)].astype(np.float32)
        cls.audio = y

        # Create output test dir if it doesn't exist
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        return

    def setUp(self):
        # Store CPU audio to GPU tensor of batch x channels x samples
        self.x = torch.zeros([BATCH_SIZE, self.audio.shape[0]])
        self.x[0, :] = torch.from_numpy(self.audio[: self.x.shape[1]])

    def _get_coreml_nn_model(self, traced_model):
        inputs = [ct.TensorType(name="input", shape=self.x.shape)]
        fc.save_coreml(traced_model, inputs, [], 32, self.OUTPUT_DIR + "/nn.mlmodel", None, "neuralnetwork")
        return ct.models.MLModel(self.OUTPUT_DIR + "/nn.mlmodel", compute_units=ct.ComputeUnit.ALL)

    def _get_coreml_mlpackage_model(self, traced_model):
        inputs = [ct.TensorType(name="input", shape=self.x.shape)]
        fc.save_coreml(traced_model, inputs, [], 16, self.OUTPUT_DIR + "/nn.mlpackage", None, "mlprogram")
        return ct.models.MLModel(self.OUTPUT_DIR + "/nn.mlpackage", compute_units=ct.ComputeUnit.ALL)

    def _get_onnx_model(self, nn_layer):
        fc.save_onnx(nn_layer, self.x, self.OUTPUT_DIR + "/nn.onnx", False)
        return ort.InferenceSession(self.OUTPUT_DIR + "/nn.onnx")

    class _TestSpec(torch.nn.Module):
        def __init__(self, type: Literal["spec", "mel"]):
            super().__init__()
            self.spec_layer = Spectrogram(
                sr=SR,
                n_fft=FFT_SIZE,
                hop_size=HOP_SIZE,
                n_mel=MEL_BANDS if type == "mel" else None)

        def forward(self, x):
            return self.spec_layer(x, power_scale=10, db=False, top_db=None)

    @unittest.skipUnless(platform.system() == "Darwin", "Only supported on macOS")
    def test_equivalence_spectrogram(self):

        destination_formats = [
            "nn",
            "mlpackage",
            # "onnx" Currently failing due to ONNX not supporting torch's new complex numbers impl
        ]
        sources = [
            self._TestSpec("spec"),
            self._TestSpec("mel")
        ]

        test_cases = itertools.product(destination_formats, sources)

        for sub_test in test_cases:
            with self.subTest(sub_test=sub_test):
                format, spec_layer = sub_test

                spec_layer.eval()

                traced_model = torch.jit.trace(spec_layer, self.x)

                # Compare outputs
                expected = spec_layer(self.x)
                atol, rtol = 0.025, 1e-2

                # Convert to CoreML and validate
                if format == "nn":
                    nn = self._get_coreml_nn_model(traced_model)
                    nn_out = nn.predict({"input": self.x.numpy()})["output"]
                    np.testing.assert_allclose(nn_out, expected, rtol=rtol, atol=atol, equal_nan=False)

                # Convert to MLPackage and validate
                if format == "mlpackage":
                    mlpackage = self._get_coreml_mlpackage_model(traced_model)
                    mlpackage_out = mlpackage.predict({"input": self.x.numpy()})["output"]
                    np.testing.assert_allclose(mlpackage_out, expected, rtol=rtol, atol=atol, equal_nan=False)

                # Convert to ONNX and validate
                if format == "onnx":
                    onnx = self._get_onnx_model(spec_layer)
                    onnx_out = onnx.run(["output"], {"input": self.x.numpy()})[0]
                    np.testing.assert_allclose(onnx_out, expected, rtol=rtol, atol=atol, equal_nan=False)


if __name__ == "__main__":
    unittest.main()
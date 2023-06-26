# Copyright 2022 Adobe. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

from setuptools import setup
from distutils.extension import Extension


from importlib.machinery import SourceFileLoader

with open("README.md") as file:
    long_description = file.read()

version = SourceFileLoader(
    "convmelspec.version", "convmelspec/version.py"
).load_module()

setup(
    name="convmelspec",
    version=version.version,
    description="Convertible Melspectrograms for On-Device Audio Machine Learning",
    author="Nicholas J. Bryan, Oriol Nieto, and Juan-Pablo Caceres",
    author_email="<Enter your email(s) here>",
    url="",
    packages=["convmelspec"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="stft, melspectrogram, melspec, onnx, coreml",
    license="",
    install_requires=[
        "numpy",
        "scipy",
        "torch",
        "soundfile",
        "coremltools>=7.0b1",
        "onnxruntime",
        "onnx",
        "torchaudio",
        "librosa"
    ],
)
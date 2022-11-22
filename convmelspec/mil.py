# Copyright 2022 Adobe. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

# import torch

from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY, register_torch_op
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.converters.mil import Builder as mb

import numpy as np


def register_stft(DFT_mode: str, eps: float=1e-16):

    if "stft" in _TORCH_OPS_REGISTRY:
        del _TORCH_OPS_REGISTRY["stft"]

    @register_torch_op
    def stft(context, node):

        # Extract stft inputs 
        ins = _get_inputs(context, node)
        input = ins[0]
        n_fft = ins[1]
        hop_length = ins[2]
        win_length = ins[3]
        window = ins[4]
        # center = ins[5]
        # normalized = ins[6]
        # onesided = ins[7]
        assert win_length.val == n_fft.val, "TODO: not implemented yet."

        # Setup FFT and hop size
        n_fft = mb.cast(x=n_fft, dtype='fp32')
        half = mb.add(x=mb.mul(x=n_fft, y=(.5)), y=1.0)
        hop_size = mb.expand_dims(x=hop_length, axes=(0,))

        # Dynamically create DFT matrix
        n = mb.range_1d(start=0.0, end=n_fft, step=1.0, name="n")
        n = mb.expand_dims(x=n, axes=(0,))

        # Apply special logic to ensure the CoreML converter doesn't convert to const or not
        if DFT_mode == 'on_the_fly':
            # Option: dynamically construct DFT matrix at runtime
            in_slice = mb.slice_by_size(x=input, begin=(0,0,), size=(1,1,))
            n = mb.add(x=n, y=mb.mul(x=in_slice, y=eps))
        else:
            # Option: Store full (half) DFT matrix in the model
            pass 

        # Construct DFT matrix
        w = mb.range_1d(start=0.0, end=half, step=1.0, name="w")
        coeff = mb.real_div(x=2.0 * np.pi, y=n_fft)
        w = mb.mul(x=w, y=coeff)

        w = mb.expand_dims(x=w, axes=(1,))
        temp = mb.matmul(x=w, y=n)
        Fr = mb.cos(x=temp)
        Fi = mb.sub(x=0.0, y=mb.sin(x=temp))

        # Apply time-window
        if window is not None:
            DFTr = mb.mul(x=window, y=Fr)
            DFTi = mb.mul(x=window, y=Fi)
        else:
            DFTr = Fr
            DFTi = Fi

        # Expand dims to be a proper conv 1d kernel
        DFTr = mb.expand_dims(x=DFTr, axes=(1,))
        DFTi = mb.expand_dims(x=DFTi, axes=(1,))

        # Conv with DFT kernels 
        input = mb.expand_dims(x=input, axes=(1,))
        real_x = mb.conv(x=input, weight=DFTr, strides=hop_size, pad_type='valid')
        real_y = mb.conv(x=input, weight=DFTi, strides=hop_size, pad_type='valid')

        # Combine stft magnitude

        # Note: Newer PyTorch stfts output complex numbers, which are not supported in coreml
        # if return_complex=False was still supported, we could stack real and imag
        # 
        # real_x = mb.expand_dims(x=real_x, axes=(3,))
        # real_y = mb.expand_dims(x=real_y, axes=(3,))
        # y = mb.stack(values=(real_x, real_y), axis=3)
        # 
        # So, instead we assume we only care about the mag and pass it through to the external spectrogram function
        y = mb.sqrt(x=mb.add(x=mb.square(x=real_x), y=mb.square(x=real_y)))

        context.add(y, node.name)

        

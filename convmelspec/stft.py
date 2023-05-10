# Copyright 2022 Adobe. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal as sig


class MelFilterbank(nn.Module):
    """Torch mel filterbank linear layer"""

    def __init__(
        self,
        sr: float,
        n_mel: int,
        n_fft: int,
        mel_mode: str = "librosa",
        fmin: float = 0.0,
        fmax: float = None,
        mel_scale="slaney",
        norm=None,
    ):

        super(MelFilterbank, self).__init__()

        # Parameters
        self.fs = sr
        self.n_mel = n_mel
        self.n_fft = n_fft
        self.half_N = int(n_fft / 2.0 + 1)

        # Mel Layer
        self.mel_analysis = nn.Linear(self.half_N, n_mel, bias=False)

        # Initialize weights with the Mel filter-bank
        if fmax == None:
            fmax = sr / 2.0

        if mel_mode == "librosa":
            import librosa  # lazy loading to allow minimal package dependencies

            if mel_scale == "slaney":
                htk = False
            elif mel_scale == "htk":
                htk = True
            mel_matrix = librosa.filters.mel(
                sr=self.fs,
                n_fft=self.n_fft,
                n_mels=self.n_mel,
                fmin=fmin,
                fmax=fmax,
                htk=htk,
            )
            self.mel_analysis.weight.data.copy_(torch.from_numpy(mel_matrix))

        elif mel_mode == "torchaudio":
            import torchaudio  # lazy loading to allow minimal package dependencies

            temp = torchaudio.transforms.MelScale(
                self.n_mel,
                self.fs,
                fmin,
                f_max=fmax,
                n_stft=int(n_fft / 2 + 1),
                norm=norm,
                mel_scale=mel_scale,
            )
            self.mel_analysis.weight.data.copy_(temp.fb.T)

            # Torchaudio 0.11
            # self.mel = torchaudio.functional.melscale_fbanks(n_fft,
            #                                                  fmin,
            #                                                  fmax,
            #                                                  n_mel,
            #                                                  sample_rate=sr,
            #                                                  norm=norm,
            #                                                  mel_scale=mel_scale)

        else:
            raise RuntimeError(
                f"Mel mode {mel_mode} is not supported (available modes "
                "are 'librosa' and 'torchaudio')"
            )

        # Make no gradient, so it doesn't train
        for param in self.mel_analysis.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor):
        """Inference

        Args:
            x (torch.tensor): input spectrogram audio (batch x FFT channels, frames)

        Returns:
            _type_: (batch x mel channels, frames)
        """
        x = torch.transpose(x, 1, 2)
        mel_x = self.mel_analysis(x)
        mel_x = torch.transpose(mel_x, 1, 2)
        return mel_x


def _Create_DFT_matrix_func(
    n: torch.Tensor, w: torch.Tensor, window: Optional[torch.Tensor] = None
):
    W = torch.unsqueeze(w, 1)
    N = torch.unsqueeze(n, 0)
    temp = torch.matmul(W, N)
    Fr = torch.cos(temp)
    Fi = -torch.sin(temp)

    # Apply time-window
    if window is not None:
        DFTr = torch.unsqueeze(torch.mul(window, Fr), 1)
        DFTi = torch.unsqueeze(torch.mul(window, Fi), 1)
    else:
        DFTr = Fr
        DFTi = Fi

    return DFTr, DFTi


class _STFT_Internal(nn.Module):
    """Short-time Fourier Transform internal class"""

    def __init__(
        self,
        dft_mode: str = "on-the-fly",
        n: Optional[torch.Tensor] = None,
        w: Optional[torch.Tensor] = None,
        window: Optional[torch.Tensor] = None,
        padding: int = 0,
    ):
        super(_STFT_Internal, self).__init__()
        self.dft_mode = dft_mode
        self._create_DFT_matrix = _Create_DFT_matrix_func
        self.no_impact = 1e-16
        self.padding = padding

        if self.dft_mode == "store":
            assert n != None
            assert w != None
            DFTr, DFTi = self._create_DFT_matrix(n, w, window=window)
            self.register_buffer("DFTr", DFTr)
            self.register_buffer("DFTi", DFTi)

    def get_DFT(self):
        if self.dft_mode == "store":
            return self.DFTr, self.DFTi
        else:
            return None, None

    def complex_to_abs(self, real_x, imag_x, power=False):
        """Convert the real and imaginary parts to magnitude or power
        spectrum."""
        S = real_x**2 + imag_x**2
        if not power:
            S = torch.sqrt(S)
        return S

    def forward_input(
        self,
        x: torch.Tensor,
        DFTr: torch.Tensor,
        DFTi: torch.Tensor,
        hop_size: int = 512,
        power: bool = True,
    ):
        # 1D Convolution separately for real and imaginary parts of DFT matrix
        real_x = F.conv1d(x, DFTr, stride=hop_size, padding=self.padding)
        imag_x = F.conv1d(x, DFTi, stride=hop_size, padding=self.padding)
        return self.complex_to_abs(real_x, imag_x, power=power)

    def forward_precomputed(
        self, x: torch.Tensor, hop_size: int = 512, power: bool = True
    ):

        # 1D Convolution separately for real and imaginary parts of DFT matrix
        real_x = F.conv1d(x, self.DFTr, stride=hop_size, padding=self.padding)
        imag_x = F.conv1d(x, self.DFTi, stride=hop_size, padding=self.padding)
        return self.complex_to_abs(real_x, imag_x, power=power)

    def forward_on_the_fly(
        self,
        x: torch.Tensor,
        n: torch.Tensor,
        w: torch.Tensor,
        hop_size: int = 512,
        power: bool = True,
        window: torch.Tensor = None,
        coreml: bool = False,
    ):

        # Only needed for Core ML
        if coreml:
            n = n + x[0, 0, 0] * self.no_impact

        DFTr, DFTi = self._create_DFT_matrix(n, w, window=window)
        return self.forward_input(x, DFTr, DFTi, hop_size, power=power)

    def forward(
        self,
        x: torch.Tensor,
        n: torch.Tensor,
        w: torch.Tensor,
        hop_size: int = 512,
        power: bool = True,
        window: Optional[torch.Tensor] = None,
    ):
        """Inference of internal STFT module

        Args:
            x (_type_): input audio signal batch x samples
            n (_type_): DFT n sequence
            w (_type_): DFT omega sequence
            n_fft (int, optional): FFT size. Defaults to 1024.
            hop_size (int, optional): Hop size. Defaults to 512.
            power (bool, optional): _description_. Defaults to True.
            window (_type_, optional): torch tensor of window. Defaults to None.

        Returns:
            _type_: spectrogram
        """
        # Dynamically compute DFT matrix every time (saves model space, wastes CPU)
        return self.forward_precomputed(
            x, self.DFTr, self.DFTi, hop_size, power=power
        )

    

class ConvertibleSpectrogram(nn.Module):
    """Convertible Spectrogram

    This layer computes a specific spectrogram that should be compatible with
    different on-device formats and bit precision. More specifically:

    - Formats:
        - ONNX
        - CoreML

    - Bit Precision:
        - Full (FP32)
        - Mixed
        - Half (FP16)

    To achieve such compatibility, this class has two different `spec_mode`s:

    - "torchaudio": This is compatible with ONNX v17 and above.
    - "DFT": This is compatible with both CoreML and ONNX.

    It should be possible to train using one mode and do inference with
    a different one (i.e., train with TorchAudio and export to CoreML using
    the DFT mode).

    For log-melspectrograms of mixed and half precision, it is recommended
    to use a `top_db` of 65dBs.
    """

    def __init__(
        self,
        sr: float = 16000,
        n_fft: int = 1024,
        window: Union[str, np.ndarray] = "hann",
        hop_size: int = 512,
        n_mel: int = None,
        spec_mode: str = "torchaudio",
        mel_mode: str = "torchaudio",
        fmin: float = 0.0,
        fmax: float = None,
        mel_scale="htk",
        norm=None,
        padding: int = 512,
        eps: float = 1e-8,
        dft_mode: str = "on_the_fly",
        coreml: bool = False,
    ):
        """_summary_

        Args:
            sr (float, optional): Sampling rate. Defaults to 16000.
            n_fft (int, optional): FFT size. Defaults to 1024.
            window (str, optional): Window torch vector or string. Defaults to 'hann'.
            hop_size (int, optional): STFT hop. Defaults to 512.
            n_mel (int, optional): number of mel bins. Defaults to None.
            spec_mode (str, optional): 'torchaudio' or 'DFT'. Defaults to 'torchaudio'.
            mel_mode (str, optional): 'librosa' or 'torchaudio'. Defaults to 'torchaudio'.
            fmin (float, optional): Min freq for mels. Defaults to 0.0.
            fmax (float, optional): Max freq for mels. Defaults to None.
            mel_scale (str, optional): 'slaney' or 'htk' api follows each separately. Defaults to 'slaney'.
            norm (_type_, optional): See librosa or torchaudio. Defaults to None.
            eps (float, optional): dB floor. Defaults to 1e-8.
            dft_mode (str, optional): 'on_the_fly', 'store', 'input'. Defaults to 'store'.
                on_the_fly = Dynamically creates DFT matrix during inference
                    model_size = pretty small
                    inference_speed = mild overhead to create DFT matrix
                    training = mild overhead to create DFT during inference calls
                    on-device integration = Easy
                store = Statically creates DFT matrix, uses precomputed matrix
                    model_size = largest
                    inference_speed = fastest
                    training = use this
                    on-device integration = Easy
                input = Returns DFT matrix for you to provide as an input
                    model_size = smallest
                    inference_speed = requires CPU-to-GPU copy of DFT matrix
                    training = unnecessary, use store
                    on-device integration = Most complicated
                Note: CoreML on_the_fly is internally optimized to store. No workaround so far.
            coreml (bool, optional): Whether to use a coreml-compatible version
        """
        super(ConvertibleSpectrogram, self).__init__()

        self.device = "cpu"
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.n_mel = n_mel
        self.py = np.pi
        self.dft_mode = dft_mode
        self.eps = eps
        self.coreml = coreml
        self.padding = padding
        # self._window = window
        self.window_fn = self._create_window_fn(window)
        self.spec_transf = None
        self.stft = None

        # Create mode (torchaudio vs. DFT and DTF mode if applicable)
        self.set_mode(spec_mode, dft_mode=dft_mode, coreml=self.coreml)

        if self.n_mel:
            self.mel = MelFilterbank(
                sr=sr,
                n_mel=n_mel,
                n_fft=n_fft,
                mel_mode=mel_mode,
                fmin=fmin,
                fmax=fmax,
                mel_scale=mel_scale,
                norm=norm,
            )

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        if "cpu" in args:
            self.device = "cpu"
        elif "cuda" in args:
            self.device = "cuda"
        return self

    def _create_window_fn(self, window: Union[str, np.ndarray, torch.Tensor]) -> Callable:
        """Creates a window function based on the input window.

        Args:
            window (str, torch.Tensor or np.ndarray): Window type or actual window in an
                ndarray.

        Returns:
            Callable: Function to create the torch window.
        """
        if type(window).__module__ == torch.__name__:

            def window_fn(win_len):
                return window.to(
                    self.device
                )

        elif type(window).__module__ == np.__name__:

            def window_fn(win_len):
                return torch.from_numpy(window.astype(np.float32)).to(
                    self.device
                )

        elif window == "hann":

            def window_fn(win_len):
                window = sig.windows.hann(win_len, sym=True)
                return torch.from_numpy(window.astype(np.float32)).to(
                    self.device
                )
                
        else:
            raise RuntimeError(
                f"Unsuported window parameter {window}. "
                "Valid options are 'hann', torch.tensor or an np.ndarray"
            )
        return window_fn

    def set_mode(self, spec_mode: str, dft_mode: str="on_the_fly", coreml: bool = False):
        """Set the DFT mode. See docs above.

        Args:
            spec_mode (str, optional): 'torchaudio' or 'DFT'. Defaults to 'torchaudio'.
            dft_mode (str): 'on_the_fly', 'store', 'input'. Defaults to 'store'.
            coreml (bool, optional): Whether to use a coreml-compatible version, only needed for on_the_fly

        Returns:
            _type_: Real DFT, Imag DFT as torch.Tensor
        """
        self.spec_mode = spec_mode

        # Type of spectrogram to use
        if self.spec_mode == "DFT":
            self.spec_transf = None
            # Create internal variables needed for DFT
            self.dft_mode = dft_mode
            self.stft = None
            self.coreml = coreml

            # Create and/or store the STFT window
            self.register_buffer("window", self.window_fn(self.n_fft))

            # Clear DFT state
            n = torch.arange(
                0,
                self.n_fft,
                dtype=torch.float32,
                requires_grad=False,
                device=self.device,
            )
            self.register_buffer("n", n)
            w = (2.0 * self.py / self.n_fft) * torch.arange(
                0,
                self.n_fft / 2 + 1,
                dtype=torch.float32,
                requires_grad=False,
                device=self.device,
            )
            self.register_buffer("w", w)

            if self.dft_mode == "on_the_fly":
                self.stft = _STFT_Internal(dft_mode=dft_mode, padding=self.padding)
                # keep w, n, window
                return None, None

            elif self.dft_mode == "store":
                self.stft = _STFT_Internal(
                    dft_mode=dft_mode,
                    n=n,
                    w=w,
                    window=self.window,
                    padding=self.padding,
                )
                # Purge w, n, window (was already turned into DFT matrix)
                self.w = None
                self.n = None
                self.window = None
                return self.stft.get_DFT()

            elif self.dft_mode == "input":
                self.stft = _STFT_Internal(dft_mode=dft_mode, padding=self.padding)
                # Purge w, n, window (will be input as DFT matrix)
                DFTr, DFTi = _Create_DFT_matrix_func(n, w, self.window)
                self.w = None
                self.n = None
                self.window = None
                return DFTr, DFTi

            else:
                raise RuntimeError(f"DFT mode f{self.dft_mode} not supported")

        elif self.spec_mode == "torchaudio":
            import torchaudio  # lazy import
            self.stft = None
            self.spec_transf = torchaudio.transforms.Spectrogram(
                n_fft=self.n_fft,
                hop_length=self.hop_size,
                power=2.0,
                center=False,
                window_fn=self.window_fn,
                pad=self.padding,
            )




    def forward(
        self,
        x: torch.Tensor,
        DFTr: Optional[torch.Tensor] = None,
        DFTi: Optional[torch.Tensor] = None,
        power: bool = True,
        db: bool = False,
        top_db: float = None,
    ):
        """_summary_

        Args:
            x (torch.Tensor): input audio batch x samples
            DFTr (torch.Tensor, optional): Real-part of DFT matrix. Defaults to None. Only used for DFT input mode.
            DFTi (torch.Tensor, optional): Imag-part of DFT matrix. Defaults to None. Only used for DFT input mode.
            power (bool, optional): power or mag. Defaults to True.
            db (bool, optional): Decibel scale or not. Defaults to False.
            top_db (float, optional): librosa style normalization. Defaults to None.

        Returns:
            _type_: tensor of (batch x mel x frames)
        """
        x = x.unsqueeze(1)  # Add channel: (batch x channel x samples)
        if self.spec_mode == "torchaudio":
            self.spec_transf.power = 2.0 if power else None
            out = self.spec_transf(x.squeeze(1))

        elif self.spec_mode == "DFT":
            with torch.cuda.amp.autocast(enabled=False):
                if self.dft_mode == "store":
                    out = self.stft.forward_precomputed(
                        x, hop_size=self.hop_size, power=power
                    )
                elif self.dft_mode == "on_the_fly":
                    out = self.stft.forward_on_the_fly(
                        x,
                        self.n,
                        self.w,
                        hop_size=self.hop_size,
                        power=power,
                        window=self.window,
                        coreml=self.coreml,
                    )
                elif self.dft_mode == "input":
                    assert DFTr != None
                    assert DFTi != None
                    out = self.stft.forward_input(
                        x, DFTr, DFTi, hop_size=self.hop_size, power=power
                    )
        else:
            raise RuntimeError(
                f"Unsupported spec_mode {self.spec_mode} "
                "(supported modes are 'torchaudio', 'DFT')"
            )

        if self.n_mel:
            out = self.mel(out)

        if db:
            scale = 10.0 if power else 20.0
            out = scale * torch.log10(out + self.eps)

            # Output shape: 1 x mel x frames
            if top_db is not None:
                thresholds = (
                    torch.max(
                        torch.max(out, dim=1, keepdim=False)[0],
                        dim=1,
                        keepdim=False,
                    )[0]
                    - top_db
                )
                out = torch.maximum(out, thresholds.reshape(-1, 1, 1))

        return out


if __name__ == "__main__":
    sample_rate=16000
    n_fft=400
    win_length=400
    hop_length=160
    n_mels=80
    coef=0.97
    fmin = 20
    fmax = 7600
    
    spec = ConvertibleSpectrogram(sr=16_000, \
                                            n_fft=n_fft,\
                                            hop_size=hop_length, \
                                            n_mel=n_mels, \
                                            fmin = fmin, fmax = fmax, \
                                            window=torch.hamming_window(win_length),
                                            spec_mode="DFT",
                                            mel_mode="torchaudio",
                                            mel_scale="htk",)
    
    print('execution success')
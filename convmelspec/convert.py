# Copyright 2022 Adobe. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.


from sys import platform
from typing import Any, Literal, Optional

import coremltools as ct
import torch
from coremltools.models.neural_network import quantization_utils


def save_onnx(model: object,
              input_tensor: torch.tensor,
              output_path: str,
              verbose: bool,
              input_names=['input'],
              output_names=['output']):
    """Save an ONNX model from a PyTorch model.
    Args:
        model (object): PyTorch module
        input_tensor (torch.tensor): input tensor to use for ONNX export function
        output_path (str): File path to save the ONNX model
        verbose (bool): Flag to be verbose or not
    """
    torch.onnx.export(model,
                input_tensor,
                output_path,
                verbose=verbose,
                input_names = input_names,
                output_names = output_names,
                dynamic_axes={'input' : {0 : 'batch_size'},
                              'output' : {0 : 'batch_size'}})


def save_coreml(traced_model: object,
                input_tensors: list,
                oshape: list,
                nbits: int,
                output_path: str,
                minimum_deployment_target: Optional[Any]=None,
                convert_to: Literal['neuralnetwork', 'mlpackage'] = 'neuralnetwork'):
    """Save a traced PyTorch model to CoreML.
    To trace a model, please follow:
        traced_model = torch.jit.trace(model, dummy_input)
        torch.jit.save(traced_model, output_torch_path)
    Parameters
    ----------
    traced_model : object
        A traced PyTorch model, see code example above
    input_tensors : list
        List of tensor shape tuples 4-tuple
    oshape : list
        Output tensor dimension tuple
    nbits : int
        Number of bits, 16 or 32-bit
    output_path : str
        File path of model output
    minimum_deployment_target : Any, optional
        Min target to deploy the model, by default None
    convert_to : Literal['mlmodel', 'mlpackage'], optional
        Convert to mlmodel or mlpackage, by default 'mlmodel'
    """
    assert len(input_tensors) > 0

    outputs = [ct.TensorType(name="output")]
    model_out = ct.convert(model=traced_model,
                         inputs=input_tensors,
                         outputs=outputs,
                         minimum_deployment_target=minimum_deployment_target,
                         convert_to=convert_to)

    # Quantize if needed
    model_specs = None
    if convert_to == 'neuralnetwork' and nbits == 16:
        model_or_specs = quantization_utils.quantize_weights(model_out, nbits=nbits)
        model_out = model_or_specs if isinstance(model_or_specs, ct.models.MLModel) else ct.models.MLModel(model_or_specs)

    model_out.save(output_path)

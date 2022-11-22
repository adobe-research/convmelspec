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
from typing import Any

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
                compute_units: ct.ComputeUnit=ct.ComputeUnit.ALL,
                minimum_deployment_target: Any=None):
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
    compute_units : ct.ComputeUnit, optional
        Type of compute unit to use, by default ct.ComputeUnit.ALL
    minimum_deployment_target : Any, optional
        Min target to deploy the model, by default None
    """
    assert len(input_tensors) > 0
    mlmodel = ct.convert(model=traced_model,
                         inputs=input_tensors,
                         compute_units=compute_units,
                         minimum_deployment_target=minimum_deployment_target)

    # Quantize if needed
    model_specs = None
    if nbits == 16:
        # The following returns a model on macOS, and only the specs otherwise
        model_or_specs = quantization_utils.quantize_weights(mlmodel,
                                                             nbits=nbits)
        if platform == "darwin":
            mlmodel = model_or_specs
        else:
            model_specs = model_or_specs
    if not model_specs:
        model_specs = mlmodel.get_spec()

    # Rename output to "output", following
    #   https://github.com/apple/coremltools/issues/775
    current_output_names = mlmodel.output_description._fd_spec
    old_name = current_output_names[0].name
    new_name = "output"
    ct.utils.rename_feature(model_specs, old_name, new_name, rename_outputs=True)

    # Set the output dimensions
    for d in oshape:
        model_specs.description.output[0].type.multiArrayType.shape.append(d)
    model_out = ct.models.MLModel(model_specs, compute_units=compute_units)
    model_out.save(output_path)

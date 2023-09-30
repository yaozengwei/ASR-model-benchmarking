#!/usr/bin/env python3
#
# Copyright 2023 Xiaomi Corporation     (Author: Zengwei Yao)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Usage: ./zipformer/profile.py
"""

import argparse
import logging
import torch

from typing import Tuple
from torch import Tensor, nn

from profiler import get_model_profile
from scaling import BiasNorm
from zipformer import BypassModule
from zipformer import get_zipformer_model


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-scale",
        type=str,
        default="large",
        help="Model scale, could be in ['large', 'medium', 'small']",
    )

    return parser.parse_args()


def _bias_norm_flops_compute(module, input, output):
    assert len(input) == 1, len(input)
    # estimate as layer_norm, see icefall/profiler.py
    flops = input[0].numel() * 5
    module.__flops__ += int(flops)


def _bypass_module_flops_compute(module, input, output):
    # For Bypass module
    assert len(input) == 2, len(input)
    flops = input[0].numel() * 2
    module.__flops__ += int(flops)


MODULE_HOOK_MAPPING = {
    BiasNorm: _bias_norm_flops_compute,
    BypassModule: _bypass_module_flops_compute,
}


@torch.no_grad()
def main():
    args = get_args()

    model, params = get_zipformer_model(scale=args.model_scale)
    num_param = sum([p.numel() for p in model.parameters()])
    print(f"Number of model parameters: {num_param}")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    logging.info(f"Device: {device}")

    model.eval()
    model.to(device)

    # for 30-second input
    B, T, D = 1, 3000, 80
    feature = torch.ones(B, T, D, dtype=torch.float32).to(device)
    feature_lens = torch.full((B,), T, dtype=torch.int64).to(device)

    flops, params = get_model_profile(
        model=model,
        args=(feature, feature_lens),
        module_hoop_mapping=MODULE_HOOK_MAPPING,
    )
    logging.info(f"For the encoder part, params: {params}, flops: {flops}")


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)

    main()

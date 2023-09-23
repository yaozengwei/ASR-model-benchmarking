# Copyright 2021, Maxime Burchi.
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

import logging
# PyTorch
import torch
import torch.nn as nn

from utils import AttributeDict

# Blocks
from models.blocks import (
    ConformerBlock
)

# Modules
from models.modules import (
    Conv1dSubsampling,
    Conv2dSubsampling,
    Conv2dPoolSubsampling,
    VGGSubsampling
)

# Positional Encodings and Masks
from models.attentions import (
    SinusoidalPositionalEncoding,
    StreamingMask
)

###############################################################################
# Encoder Models
###############################################################################


class ConformerEncoder(nn.Module):

    def __init__(self, params):
        super(ConformerEncoder, self).__init__()

        # Subsampling Module
        if params["subsampling_module"] == "Conv1d":
            self.subsampling_module = Conv1dSubsampling(params["subsampling_layers"], params["n_mels"], params["subsampling_filters"], params["subsampling_kernel_size"], params["subsampling_norm"], params["subsampling_act"])
        elif params["subsampling_module"] == "Conv2d":
            self.subsampling_module = Conv2dSubsampling(params["subsampling_layers"], params["subsampling_filters"], params["subsampling_kernel_size"], params["subsampling_norm"], params["subsampling_act"])
        elif params["subsampling_module"] == "Conv2dPool":
            self.subsampling_module = Conv2dPoolSubsampling(params["subsampling_layers"], params["subsampling_filters"], params["subsampling_kernel_size"], params["subsampling_norm"], params["subsampling_act"])
        elif params["subsampling_module"] == "VGG":
            self.subsampling_module = VGGSubsampling(params["subsampling_layers"], params["subsampling_filters"], params["subsampling_kernel_size"], params["subsampling_norm"], params["subsampling_act"])
        else:
            raise Exception("Unknown subsampling module:", params["subsampling_module"])

        # Padding Mask
        self.padding_mask = StreamingMask(left_context=params.get("left_context", params["max_pos_encoding"]), right_context=0 if params.get("causal", False) else params.get("right_context", params["max_pos_encoding"]))

        # Linear Proj
        self.linear = nn.Linear(params["subsampling_filters"][-1] * params["n_mels"] // 2**params["subsampling_layers"], params["dim_model"][0] if isinstance(params["dim_model"], list) else  params["dim_model"])

        # Dropout
        self.dropout = nn.Dropout(p=params["Pdrop"])

        # Sinusoidal Positional Encodings
        self.pos_enc = None if params["relative_pos_enc"] else SinusoidalPositionalEncoding(params["max_pos_encoding"], params["dim_model"][0] if isinstance(params["dim_model"], list) else  params["dim_model"])

        # Conformer Blocks
        self.blocks = nn.ModuleList([ConformerBlock(
            dim_model=params["dim_model"][(block_id > torch.tensor(params.get("expand_blocks", []))).sum()] if isinstance(params["dim_model"], list) else params["dim_model"],
            dim_expand=params["dim_model"][(block_id >= torch.tensor(params.get("expand_blocks", []))).sum()] if isinstance(params["dim_model"], list) else params["dim_model"],
            ff_ratio=params["ff_ratio"],
            num_heads=params["num_heads"][(block_id > torch.tensor(params.get("expand_blocks", []))).sum()] if isinstance(params["num_heads"], list) else params["num_heads"],
            kernel_size=params["kernel_size"][(block_id >= torch.tensor(params.get("expand_blocks", []))).sum()] if isinstance(params["kernel_size"], list) else params["kernel_size"],
            att_group_size=params["att_group_size"][(block_id > torch.tensor(params.get("strided_blocks", []))).sum()] if isinstance(params.get("att_group_size", 1), list) else params.get("att_group_size", 1),
            att_kernel_size=params["att_kernel_size"][(block_id > torch.tensor(params.get("strided_layers", []))).sum()] if isinstance(params.get("att_kernel_size", None), list) else params.get("att_kernel_size", None),
            linear_att=params.get("linear_att", False),
            Pdrop=params["Pdrop"],
            relative_pos_enc=params["relative_pos_enc"],
            max_pos_encoding=params["max_pos_encoding"] // params.get("stride", 2)**int((block_id > torch.tensor(params.get("strided_blocks", []))).sum()),
            conv_stride=(params["conv_stride"][(block_id > torch.tensor(params.get("strided_blocks", []))).sum()] if isinstance(params["conv_stride"], list) else params["conv_stride"]) if block_id in params.get("strided_blocks", []) else 1,
            att_stride=(params["att_stride"][(block_id > torch.tensor(params.get("strided_blocks", []))).sum()] if isinstance(params["att_stride"], list) else params["att_stride"]) if block_id in params.get("strided_blocks", []) else 1,
            causal=params.get("causal", False)
        ) for block_id in range(params["num_blocks"])])

    def forward(self, x, x_len=None):

        # Subsampling Module
        x, x_len = self.subsampling_module(x, x_len)

        # Padding Mask
        mask = self.padding_mask(x, x_len)

        # Transpose (B, D, T) -> (B, T, D)
        x = x.transpose(1, 2)

        # Linear Projection
        x = self.linear(x)

        # Dropout
        x = self.dropout(x)

        # Sinusoidal Positional Encodings
        if self.pos_enc is not None:
            x = x + self.pos_enc(x.size(0), x.size(1))

        # Conformer Blocks
        attentions = []
        for block in self.blocks:
            x, attention, hidden = block(x, mask)
            attentions.append(attention)

            # Strided Block
            if block.stride > 1:

                # Stride Mask (B, 1, T // S, T // S)
                if mask is not None:
                    mask = mask[:, :, ::block.stride, ::block.stride]

                # Update Seq Lengths
                if x_len is not None:
                    x_len = torch.div(x_len - 1, block.stride, rounding_mode='floor') + 1

        return x, x_len, attentions


def model_params(scale="large") -> AttributeDict:
    large = AttributeDict(
        {
            "n_mels": 80,
            "arch": "Conformer",
            "num_blocks": 16,
            "dim_model": [360, 512, 720],
            "ff_ratio": 4,
            "num_heads": 8,
            "kernel_size": 15,
            "Pdrop": 0.1,
            "conv_stride": 2,
            "att_stride": 1,
            "strided_blocks": [4, 10],
            "expand_blocks": [4, 10],
            "att_group_size": [3, 1, 1],
            "relative_pos_enc": True,
            "max_pos_encoding": 10000,
            "subsampling_module": "Conv2d",
            "subsampling_layers": 1,
            "subsampling_filters": [360],
            "subsampling_kernel_size": 3,
            "subsampling_norm": "batch",
            "subsampling_act": "swish",
        }
    )

    medium = AttributeDict(
        {
            "n_mels": 80,
            "arch": "Conformer",
            "num_blocks": 16,
            "dim_model": [180, 256, 360],
            "ff_ratio": 4,
            "num_heads": 4,
            "kernel_size": 15,
            "Pdrop": 0.1,
            "conv_stride": 2,
            "att_stride": 1,
            "strided_blocks": [4, 10],
            "expand_blocks": [4, 10],
            "att_group_size": [3, 1, 1],
            "relative_pos_enc": True,
            "max_pos_encoding": 10000,
            "subsampling_module": "Conv2d",
            "subsampling_layers": 1,
            "subsampling_filters": [180],
            "subsampling_kernel_size": 3,
            "subsampling_norm": "batch",
            "subsampling_act": "swish",
        }
    )

    small = AttributeDict(
        {
            "n_mels": 80,
            "arch": "Conformer",
            "num_blocks": 15,
            "dim_model": [120, 168, 240],
            "ff_ratio": 4,
            "num_heads": 4,
            "kernel_size": 15,
            "Pdrop": 0.1,
            "conv_stride": 2,
            "att_stride": 1,
            "strided_blocks": [4, 9],
            "expand_blocks": [4, 9],
            "att_group_size": [3, 1, 1],
            "relative_pos_enc": True,
            "max_pos_encoding": 10000,
            "subsampling_module": "Conv2d",
            "subsampling_layers": 1,
            "subsampling_filters": [120],
            "subsampling_kernel_size": 3,
            "subsampling_norm": "batch",
            "subsampling_act": "swish",
        }
    )

    params = {"large": large, "medium": medium, "small": small}
    return params[scale]


def get_efficient_conformer_model(scale="large") -> nn.Module:
    params = model_params(scale)
    model = ConformerEncoder(params)
    return model, params


def _test_efficient_conformer_main(scale="large"):
    model, params = get_efficient_conformer_model(scale)

    model.eval()

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    batch_size = 5
    seq_len = 20
    # Just make sure the forward pass runs.
    f = model(
        torch.randn(batch_size, params.n_mels, seq_len),
        torch.full((batch_size,), seq_len, dtype=torch.int64),
    )
    f  # to remove flake8 warnings


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _test_efficient_conformer_main(scale="large")
    _test_efficient_conformer_main(scale="medium")
    _test_efficient_conformer_main(scale="small")



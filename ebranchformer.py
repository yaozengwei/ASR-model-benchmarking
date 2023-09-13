import logging
import torch
from torch import nn

from espnet2.asr.encoder.e_branchformer_encoder import EBranchformerEncoder
from utils import AttributeDict

"""
Refer to
https://arxiv.org/pdf/2210.00077.pdf and
https://github.com/espnet/espnet/blob/master/egs2/librispeech/asr1/conf/tuning/train_asr_e_branchformer.yaml
"""


def model_params(scale="large") -> AttributeDict:
    large = AttributeDict(
        {
            "input_size": 80,
            "output_size": 512,
            "attention_heads": 8,
            "attention_layer_type": "rel_selfattn",
            "pos_enc_layer_type": "rel_pos",
            "rel_pos_type": "latest",
            "cgmlp_linear_units": 3072,
            "cgmlp_conv_kernel": 31,
            "use_linear_after_conv": False,
            "gate_activation": "identity",
            "num_blocks": 17,
            "dropout_rate": 0.1,
            "positional_dropout_rate": 0.1,
            "attention_dropout_rate": 0.1,
            "input_layer": "conv2d",
            "layer_drop_rate": 0.1,
            "linear_units": 1024,
            "positionwise_layer_type": "linear",
            "macaron_ffn": True,
            "use_ffn": True,
            "merge_conv_kernel": 31,
        }
    )
    base = AttributeDict(
        {
            "input_size": 80,
            "output_size": 256,
            "attention_heads": 4,
            "attention_layer_type": "rel_selfattn",
            "pos_enc_layer_type": "rel_pos",
            "rel_pos_type": "latest",
            "cgmlp_linear_units": 1536,
            "cgmlp_conv_kernel": 31,
            "use_linear_after_conv": False,
            "gate_activation": "identity",
            "num_blocks": 16,
            "dropout_rate": 0.1,
            "positional_dropout_rate": 0.1,
            "attention_dropout_rate": 0.1,
            "input_layer": "conv2d",
            "layer_drop_rate": 0.1,
            "linear_units": 512,
            "positionwise_layer_type": "linear",
            "macaron_ffn": True,
            "use_ffn": True,
            "merge_conv_kernel": 31,
        }
    )

    params = {"large": large, "base": base}
    return params[scale]


def get_ebranchformer_model(scale="large") -> nn.Module:
    params = model_params(scale)
    # TODO: We can add an option to switch between Conformer and Transformer
    model = EBranchformerEncoder(
        input_size=params.input_size,
        output_size=params.output_size,
        attention_heads=params.attention_heads,
        attention_layer_type=params.attention_layer_type,
        pos_enc_layer_type=params.pos_enc_layer_type,
        rel_pos_type=params.rel_pos_type,
        cgmlp_linear_units=params.cgmlp_linear_units,
        cgmlp_conv_kernel=params.cgmlp_conv_kernel,
        use_linear_after_conv=params.use_linear_after_conv,
        gate_activation=params.gate_activation,
        num_blocks=params.num_blocks,
        dropout_rate=params.dropout_rate,
        positional_dropout_rate=params.positional_dropout_rate,
        attention_dropout_rate=params.attention_dropout_rate,
        input_layer=params.input_layer,
        layer_drop_rate=params.layer_drop_rate,
        linear_units=params.linear_units,
        positionwise_layer_type=params.positionwise_layer_type,
        macaron_ffn=params.macaron_ffn,
        use_ffn=params.use_ffn,
        merge_conv_kernel=params.merge_conv_kernel,
    )
    return model, params


def _test_ebranchformer_main(scale="large"):
    model, params = get_ebranchformer_model(scale)

    model.eval()

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    batch_size = 5
    seq_len = 20
    # Just make sure the forward pass runs.
    f = model(
        torch.randn(batch_size, seq_len, params.input_size),
        torch.full((batch_size,), seq_len, dtype=torch.int64),
    )
    f  # to remove flake8 warnings


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _test_ebranchformer_main(scale="large")
    _test_ebranchformer_main(scale="base")



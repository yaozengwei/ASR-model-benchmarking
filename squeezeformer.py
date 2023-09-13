import logging
import torch
from torch import Tensor, nn

from nemo.collections.asr.modules.squeezeformer_encoder import SqueezeformerEncoder
from utils import AttributeDict


def model_params(scale) -> AttributeDict:
    extra_small = AttributeDict(
        {
            "feat_in": 80,
            "feat_out": -1,
            "n_layers": 16,
            "d_model": 144,
            "adaptive_scale": True,
            "time_reduce_idx": 7,
            "time_recovery_idx": None,
            "subsampling": "dw_striding",
            "subsampling_factor": 4,
            "subsampling_conv_channels": -1,
            "ff_expansion_factor": 4,
            "self_attention_model": "rel_pos",
            "n_heads": 4,
            "att_context_size": [-1, -1],
            "xscaling": True,
            "untie_biases": True,
            "pos_emb_max_len": 5000,
            "conv_kernel_size": 31,
            "conv_norm_type": "batch_norm",
            "dropout": 0.1,
            "dropout_emb": 0.0,
            "dropout_att": 0.1,
        }
    )
    small = AttributeDict(
        {
            "feat_in": 80,
            "feat_out": -1,
            "n_layers": 18,
            "d_model": 196,
            "adaptive_scale": True,
            "time_reduce_idx": 8,
            "time_recovery_idx": None,
            "subsampling": "dw_striding",
            "subsampling_factor": 4,
            "subsampling_conv_channels": -1,
            "ff_expansion_factor": 4,
            "self_attention_model": "rel_pos",
            "n_heads": 4,
            "att_context_size": [-1, -1],
            "xscaling": True,
            "untie_biases": True,
            "pos_emb_max_len": 5000,
            "conv_kernel_size": 31,
            "conv_norm_type": "batch_norm",
            "dropout": 0.1,
            "dropout_emb": 0.0,
            "dropout_att": 0.1,
        }
    )
    small_medium = AttributeDict(
        {
            "feat_in": 80,
            "feat_out": -1,
            "n_layers": 16,
            "d_model": 256,
            "adaptive_scale": True,
            "time_reduce_idx": 7,
            "time_recovery_idx": None,
            "subsampling": "dw_striding",
            "subsampling_factor": 4,
            "subsampling_conv_channels": -1,
            "ff_expansion_factor": 4,
            "self_attention_model": "rel_pos",
            "n_heads": 4,
            "att_context_size": [-1, -1],
            "xscaling": True,
            "untie_biases": True,
            "pos_emb_max_len": 5000,
            "conv_kernel_size": 31,
            "conv_norm_type": "batch_norm",
            "dropout": 0.1,
            "dropout_emb": 0.0,
            "dropout_att": 0.1,
        }
    )
    medium = AttributeDict(
        {
            "feat_in": 80,
            "feat_out": -1,
            "n_layers": 20,
            "d_model": 324,
            "adaptive_scale": True,
            "time_reduce_idx": 9,
            "time_recovery_idx": None,
            "subsampling": "dw_striding",
            "subsampling_factor": 4,
            "subsampling_conv_channels": -1,
            "ff_expansion_factor": 4,
            "self_attention_model": "rel_pos",
            "n_heads": 4,
            "att_context_size": [-1, -1],
            "xscaling": True,
            "untie_biases": True,
            "pos_emb_max_len": 5000,
            "conv_kernel_size": 31,
            "conv_norm_type": "batch_norm",
            "dropout": 0.1,
            "dropout_emb": 0.0,
            "dropout_att": 0.1,
        }
    )
    medium_large = AttributeDict(
        {
            "feat_in": 80,
            "feat_out": -1,
            "n_layers": 18,
            "d_model": 512,
            "adaptive_scale": True,
            "time_reduce_idx": 8,
            "time_recovery_idx": None,
            "subsampling": "dw_striding",
            "subsampling_factor": 4,
            "subsampling_conv_channels": -1,
            "ff_expansion_factor": 4,
            "self_attention_model": "rel_pos",
            "n_heads": 8,
            "att_context_size": [-1, -1],
            "xscaling": True,
            "untie_biases": True,
            "pos_emb_max_len": 5000,
            "conv_kernel_size": 31,
            "conv_norm_type": "batch_norm",
            "dropout": 0.1,
            "dropout_emb": 0.0,
            "dropout_att": 0.1,
        }
    )
    large = AttributeDict(
        {
            "feat_in": 80,
            "feat_out": -1,
            "n_layers": 22,
            "d_model": 640,
            "adaptive_scale": True,
            "time_reduce_idx": 10,
            "time_recovery_idx": None,
            "subsampling": "dw_striding",
            "subsampling_factor": 4,
            "subsampling_conv_channels": -1,
            "ff_expansion_factor": 4,
            "self_attention_model": "rel_pos",
            "n_heads": 8,
            "att_context_size": [-1, -1],
            "xscaling": True,
            "untie_biases": True,
            "pos_emb_max_len": 5000,
            "conv_kernel_size": 31,
            "conv_norm_type": "batch_norm",
            "dropout": 0.1,
            "dropout_emb": 0.0,
            "dropout_att": 0.1,
        }
    )
    params = {
        "extra_small": extra_small,
        "small": small,
        "small_medium": small_medium,
        "medium": medium,
        "medium_large": medium_large,
        "large": large,
    }
    return params[scale]


def get_squeezeformer_model(scale) -> nn.Module:
    params = model_params(scale)
    # TODO: We can add an option to switch between Conformer and Transformer
    model = SqueezeformerEncoder(
        feat_in=params.feat_in,
        feat_out=params.feat_out,
        n_layers=params.n_layers,
        d_model=params.d_model,
        adaptive_scale=params.adaptive_scale,
        time_reduce_idx=params.time_reduce_idx,
        time_recovery_idx=params.time_recovery_idx,
        subsampling=params.subsampling,
        subsampling_factor=params.subsampling_factor,
        subsampling_conv_channels=params.subsampling_conv_channels,
        ff_expansion_factor=params.ff_expansion_factor,
        self_attention_model=params.self_attention_model,
        n_heads=params.n_heads,
        att_context_size=params.att_context_size,
        xscaling=params.xscaling,
        untie_biases=params.untie_biases,
        pos_emb_max_len=params.pos_emb_max_len,
        conv_kernel_size=params.conv_kernel_size,
        conv_norm_type=params.conv_norm_type,
        dropout=params.dropout,
        dropout_emb=params.dropout_emb,
        dropout_att=params.dropout_att,
    )
    return model, params


def _test_squeezeformer_main(scale):
    model, params = get_squeezeformer_model(scale)

    model.eval()

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    batch_size = 5
    seq_len = 20
    # Just make sure the forward pass runs.
    f = model(
        audio_signal=torch.randn(batch_size, params.feat_in, seq_len),
        length=torch.full((batch_size,), seq_len, dtype=torch.int64),
    )
    f  # to remove flake8 warnings


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _test_squeezeformer_main(scale="extra_small")
    _test_squeezeformer_main(scale="small")
    _test_squeezeformer_main(scale="small_medium")
    _test_squeezeformer_main(scale="medium")
    _test_squeezeformer_main(scale="large")






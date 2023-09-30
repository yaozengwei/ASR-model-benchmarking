#!/usr/bin/env python3
#
# Copyright    2023  Xiaomi Corp.        (authors: Zengwei Yao)
#
# See ../LICENSE for clarification regarding multiple authors
#
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

import argparse

import torch
from torch.profiler import ProfilerActivity, record_function

from utils import (
    ShapeGenerator,
    SortedShapeGenerator,
    generate_data,
    str2bool,
)

from ebranchformer import get_ebranchformer_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sort-utterance",
        type=str2bool,
        default=True,
        help="True to sort utterance duration before batching them up",
    )

    parser.add_argument(
        "--model-scale",
        type=str,
        default="large",
        help="Model scale, could be 'large' or 'base'",
    )

    return parser.parse_args()


@torch.no_grad()
def main():
    args = get_args()

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    print(f"device: {device}")

    if args.sort_utterance:
        max_frames = 100000
        suffix = f"ebranchformer-{args.model_scale}-max-frames-{max_frames}"
    else:
        # won't OOM when it's 50. Set it to 30 as torchaudio is using 30
        batch_size = 30
        suffix = f"ebranchformer-{args.model_scale}-{batch_size}"

    model, params = get_ebranchformer_model(args.model_scale)
    num_param = sum([p.numel() for p in model.parameters()])
    print(f"Number of model parameters: {num_param}")

    model.to(device)
    model.eval()

    if args.sort_utterance:
        shape_generator = SortedShapeGenerator(max_frames)
    else:
        shape_generator = ShapeGenerator(batch_size)

    print(f"Benchmarking started (Sort utterance {args.sort_utterance})")

    prof = torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=10, warmup=10, active=20, repeat=2
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            f"./log_models/{suffix}"
        ),
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    )

    prof.start()

    # for 30-second input
    B, T, D = 30, 3000, 80

    for i, shape_info in enumerate(shape_generator):
        print("i", i)

        encoder_in = torch.ones(B, T, D, dtype=torch.float32, device=device)
        encoder_in_lens = torch.full((B,), T, dtype=torch.int64, device=device)

        with record_function(suffix):
            encoder_out, encoder_out_lengths, _ = model(encoder_in, encoder_in_lens)

        if i > 80:
            break

        prof.step()
    prof.stop()
    print("Benchmarking done")

    s = str(
        prof.key_averages(group_by_stack_n=10).table(
            sort_by="self_cuda_time_total", row_limit=8
        )
    )

    with open(f"{suffix}.txt", "w") as f:
        f.write(s + "\n")


if __name__ == "__main__":
    torch.manual_seed(20220227)
    main()

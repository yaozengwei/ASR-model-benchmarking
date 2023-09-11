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

from zipformer import get_zipformer_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sort-utterance",
        type=str2bool,
        help="True to sort utterance duration before batching them up",
    )

    return parser.parse_args()


def main():
    args = get_args()

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    print(f"device: {device}")

    if args.sort_utterance:
        max_frames = 10000
        suffix = f"max-frames-{max_frames}"
    else:
        # won't OOM when it's 50. Set it to 30 as torchaudio is using 30
        batch_size = 30
        suffix = batch_size

    model, params = get_zipformer_model(scale="medium")
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
            f"./log/zipformer-{suffix}"
        ),
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    )

    prof.start()

    for i, shape_info in enumerate(shape_generator):
        print("i", i)
        (
            encoder_in,
            encoder_in_lens,
        ) = generate_data(
            shape_info,
            encoder_in_dim=params.feature_dim,  # 80
            device=device,
        )
        encoder_in_lens = encoder_in_lens.to(torch.int64)

        with record_function("zipformer"):
            encoder_out, encoder_out_lengths = model(encoder_in, encoder_in_lens)

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

    with open(f"zipformer-{suffix}.txt", "w") as f:
        f.write(s + "\n")


if __name__ == "__main__":
    torch.manual_seed(20220227)
    main()

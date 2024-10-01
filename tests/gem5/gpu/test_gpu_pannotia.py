# Copyright (c) 2024 The Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import gzip
import os.path
import shutil
from urllib.request import urlretrieve

from testlib import *

resource_path = joinpath(absdirpath(__file__), "..", "gpu-pannotia-resources")
binary_path = joinpath(resource_path, "pannotia-bins")
dataset_path = joinpath(resource_path, "pannotia-datasets")

binary_links = {
    "bc.gem5": "https://storage.googleapis.com/dist.gem5.org/dist/v24-0/test-progs/pannotia/bc.gem5",
    "color_max.gem5": "https://storage.googleapis.com/dist.gem5.org/dist/v24-0/test-progs/pannotia/color_max.gem5",
    "color_maxmin.gem5": "https://storage.googleapis.com/dist.gem5.org/dist/v24-0/test-progs/pannotia/color_maxmin.gem5",
    "fw_hip.gem5": "https://storage.googleapis.com/dist.gem5.org/dist/v24-0/test-progs/pannotia/fw_hip.gem5",
    "mis_hip.gem5": "https://storage.googleapis.com/dist.gem5.org/dist/v24-0/test-progs/pannotia/mis_hip.gem5",
    "pagerank.gem5": "https://storage.googleapis.com/dist.gem5.org/dist/v24-0/test-progs/pannotia/pagerank.gem5",
    "pagerank_spmv.gem5": "https://storage.googleapis.com/dist.gem5.org/dist/v24-0/test-progs/pannotia/pagerank_spmv.gem5",
    "sssp.gem5": "https://storage.googleapis.com/dist.gem5.org/dist/v24-0/test-progs/pannotia/sssp.gem5",
    "sssp_ell.gem5": "https://storage.googleapis.com/dist.gem5.org/dist/v24-0/test-progs/pannotia/sssp_ell.gem5",
}

dataset_links = {
    "1k_128k.gr": "https://storage.googleapis.com/dist.gem5.org/dist/develop/datasets/pannotia/bc/1k_128k.gr",
    "2k_1M.gr": "https://storage.googleapis.com/dist.gem5.org/dist/develop/datasets/pannotia/bc/2k_1M.gr",
    "G3_circuit.graph": "https://storage.googleapis.com/dist.gem5.org/dist/develop/datasets/pannotia/color/G3_circuit.graph",
    "ecology1.graph": "https://storage.googleapis.com/dist.gem5.org/dist/develop/datasets/pannotia/color/ecology1.graph",
    "256_16384.gr": "https://storage.googleapis.com/dist.gem5.org/dist/develop/datasets/pannotia/floydwarshall/256_16384.gr",
    "512_65536.gr": "https://storage.googleapis.com/dist.gem5.org/dist/develop/datasets/pannotia/floydwarshall/512_65536.gr",
    "G3_circuit.graph": "https://storage.googleapis.com/dist.gem5.org/dist/develop/datasets/pannotia/mis/G3_circuit.graph",
    "ecology1.graph": "https://storage.googleapis.com/dist.gem5.org/dist/develop/datasets/pannotia/mis/ecology1.graph",
    "coAuthorsDBLP.graph": "https://storage.googleapis.com/dist.gem5.org/dist/develop/datasets/pannotia/pagerank/coAuthorsDBLP.graph",
    "USA-road-d.NY.gr.gz": "http://www.diag.uniroma1.it/challenge9/data/USA-road-d/USA-road-d.NY.gr.gz",
}


if not os.path.isdir(resource_path):
    os.makedirs(binary_path)
    os.makedirs(dataset_path)

    for name in binary_links.keys():
        urlretrieve(binary_links[name], f"{binary_path}/{name}")
    for name in dataset_links.keys():
        urlretrieve(dataset_links[name], f"{dataset_path}/{name}")

    with gzip.open(f"{dataset_path}/USA-road-d.NY.gr.gz", "rb") as f_in:
        with open(f"{dataset_path}/USA-road-d.NY.gr", "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(f"{dataset_path}/USA-road-d.NY.gr.gz")

if len(os.listdir(binary_path)) < len(binary_links):
    testlib.log.test_log.warn(
        "One or more binaries for the Pannotia GPU tests are missing! Try deleting gpu-pannotia-resources and rerunning."
    )
if len(os.listdir(dataset_path)) < len(dataset_links):
    testlib.log.test_log.warn(
        "One or more dataset files for the Pannotia GPU tests are missing! Try deleting gpu-pannotia-resources and rerunning."
    )


gem5_verify_config(
    name="gpu-apu-se-pannotia-bc-1k-128k",
    fixtures=(),
    verifiers=(),
    config=joinpath(config.base_dir, "configs", "example", "apu_se.py"),
    config_args=[
        "-n3",
        "--mem-size=8GB",
        "-c",
        joinpath(binary_path, "bc.gem5"),
        "--options",
        joinpath(dataset_path, "1k_128k.gr"),
    ],
    valid_isas=(constants.vega_x86_tag,),
    valid_hosts=(constants.host_gcn_gpu_tag,),
    length=constants.very_long_tag,
)

gem5_verify_config(
    name="gpu-apu-se-pannotia-bc-2k-1M",
    fixtures=(),
    verifiers=(),
    config=joinpath(config.base_dir, "configs", "example", "apu_se.py"),
    config_args=[
        "-n3",
        "--mem-size=8GB",
        "-c",
        joinpath(binary_path, "bc.gem5"),
        "--options",
        joinpath(dataset_path, "2k_1M.gr"),
    ],
    valid_isas=(constants.vega_x86_tag,),
    valid_hosts=(constants.host_gcn_gpu_tag,),
    length=constants.very_long_tag,
)

gem5_verify_config(
    name="gpu-apu-se-pannotia-color-maxmin-ecology",
    fixtures=(),
    verifiers=(),
    config=joinpath(config.base_dir, "configs", "example", "apu_se.py"),
    config_args=[
        "-n3",
        "--mem-size=8GB",
        "-c",
        joinpath(binary_path, "color_maxmin.gem5"),
        "--options",
        f'{joinpath(dataset_path, "ecology1.graph")} 0',
    ],
    valid_isas=(constants.vega_x86_tag,),
    valid_hosts=(constants.host_gcn_gpu_tag,),
    length=constants.very_long_tag,
)

gem5_verify_config(
    name="gpu-apu-se-pannotia-color-maxmin-g3-circuit",
    fixtures=(),
    verifiers=(),
    config=joinpath(config.base_dir, "configs", "example", "apu_se.py"),
    config_args=[
        "-n3",
        "--mem-size=8GB",
        "-c",
        joinpath(binary_path, "color_maxmin.gem5"),
        "--options",
        f'{joinpath(dataset_path, "G3_circuit.graph")} 0',
    ],
    valid_isas=(constants.vega_x86_tag,),
    valid_hosts=(constants.host_gcn_gpu_tag,),
    length=constants.very_long_tag,
)

gem5_verify_config(
    name="gpu-apu-se-pannotia-color-max-ecology",
    fixtures=(),
    verifiers=(),
    config=joinpath(config.base_dir, "configs", "example", "apu_se.py"),
    config_args=[
        "-n3",
        "--mem-size=8GB",
        "-c",
        joinpath(binary_path, "color_max.gem5"),
        "--options",
        f'{joinpath(dataset_path, "ecology1.graph")} 0',
    ],
    valid_isas=(constants.vega_x86_tag,),
    valid_hosts=(constants.host_gcn_gpu_tag,),
    length=constants.very_long_tag,
)

gem5_verify_config(
    name="gpu-apu-se-pannotia-color-max-g3-circuit",
    fixtures=(),
    verifiers=(),
    config=joinpath(config.base_dir, "configs", "example", "apu_se.py"),
    config_args=[
        "-n3",
        "--mem-size=8GB",
        "-c",
        joinpath(binary_path, "color_max.gem5"),
        "--options",
        f'{joinpath(dataset_path, "G3_circuit.graph")} 0',
    ],
    valid_isas=(constants.vega_x86_tag,),
    valid_hosts=(constants.host_gcn_gpu_tag,),
    length=constants.very_long_tag,
)

gem5_verify_config(
    name="gpu-apu-se-pannotia-fw-hip-256",
    fixtures=(),
    verifiers=(),
    config=joinpath(config.base_dir, "configs", "example", "apu_se.py"),
    config_args=[
        "-n3",
        "--mem-size=8GB",
        "-c",
        joinpath(binary_path, "fw_hip.gem5"),
        "--options",
        f'-f {joinpath(dataset_path, "256_16384.gr")} -m default',
    ],
    valid_isas=(constants.vega_x86_tag,),
    valid_hosts=(constants.host_gcn_gpu_tag,),
    length=constants.very_long_tag,
)

gem5_verify_config(
    name="gpu-apu-se-pannotia-fw-hip-512",
    fixtures=(),
    verifiers=(),
    config=joinpath(config.base_dir, "configs", "example", "apu_se.py"),
    config_args=[
        "-n3",
        "--mem-size=8GB",
        "-c",
        joinpath(binary_path, "fw_hip.gem5"),
        "--options",
        f'-f {joinpath(dataset_path, "512_65536.gr")} -m default',
    ],
    valid_isas=(constants.vega_x86_tag,),
    valid_hosts=(constants.host_gcn_gpu_tag,),
    length=constants.very_long_tag,
)

# This test fails with
# ERROR: hipMalloc row_d (size:-202182160) => hipErrorOutOfMemory
# even when mem-size is set to 64GiB.
# gem5_verify_config(
#     name="gpu-apu-se-pannotia-mis-hip-ecology",
#     fixtures=(),
#     verifiers=(),
#     config=joinpath(config.base_dir, "configs", "example", "apu_se.py"),
#     config_args=[
#         "-n3",
#         "--mem-size=64GiB",
#         "-c",
#         joinpath(binary_path, "mis_hip.gem5"),
#         "--options",
#         f'{joinpath(dataset_path, "ecology1.graph")} 0'
#     ],
#     valid_isas=(constants.vega_x86_tag,),
#     valid_hosts=(constants.host_gcn_gpu_tag,),
#     length=constants.very_long_tag,
# )

# This test also fails, likely because it also runs out of memory, but
# this hasn't been closely investigated.
# gem5_verify_config(
#     name="gpu-apu-se-pannotia-mis-hip-g3-circuit",
#     fixtures=(),
#     verifiers=(),
#     config=joinpath(config.base_dir, "configs", "example", "apu_se.py"),
#     config_args=[
#         "-n3",
#         "--mem-size=32GiB",
#         "-c",
#         joinpath(binary_path, "mis_hip.gem5"),
#         "--options",
#         f'{joinpath(dataset_path, "G3_circuit.graph")} 0'
#     ],
#     valid_isas=(constants.vega_x86_tag,),
#     valid_hosts=(constants.host_gcn_gpu_tag,),
#     length=constants.very_long_tag,
# )

gem5_verify_config(
    name="gpu-apu-se-pannotia-pagerank-spmv",
    fixtures=(),
    verifiers=(),
    config=joinpath(config.base_dir, "configs", "example", "apu_se.py"),
    config_args=[
        "-n3",
        "--mem-size=8GB",
        "-c",
        joinpath(binary_path, "pagerank_spmv.gem5"),
        "--options",
        f'{joinpath(dataset_path, "coAuthorsDBLP.graph")} 0',
    ],
    valid_isas=(constants.vega_x86_tag,),
    valid_hosts=(constants.host_gcn_gpu_tag,),
    length=constants.very_long_tag,
)

gem5_verify_config(
    name="gpu-apu-se-pannotia-pagerank",
    fixtures=(),
    verifiers=(),
    config=joinpath(config.base_dir, "configs", "example", "apu_se.py"),
    config_args=[
        "-n3",
        "--mem-size=8GB",
        "-c",
        joinpath(binary_path, "pagerank.gem5"),
        "--options",
        f'{joinpath(dataset_path, "coAuthorsDBLP.graph")} 0',
    ],
    valid_isas=(constants.vega_x86_tag,),
    valid_hosts=(constants.host_gcn_gpu_tag,),
    length=constants.very_long_tag,
)

gem5_verify_config(
    name="gpu-apu-se-pannotia-sssp-ell",
    fixtures=(),
    verifiers=(),
    config=joinpath(config.base_dir, "configs", "example", "apu_se.py"),
    config_args=[
        "-n3",
        "--mem-size=8GB",
        "-c",
        joinpath(binary_path, "sssp_ell.gem5"),
        "--options",
        f'{joinpath(dataset_path, "USA-road-d.NY.gr")} 0',
    ],
    valid_isas=(constants.vega_x86_tag,),
    valid_hosts=(constants.host_gcn_gpu_tag,),
    length=constants.very_long_tag,
)


gem5_verify_config(
    name="gpu-apu-se-pannotia-sssp",
    fixtures=(),
    verifiers=(),
    config=joinpath(config.base_dir, "configs", "example", "apu_se.py"),
    config_args=[
        "-n3",
        "--mem-size=8GB",
        "-c",
        joinpath(binary_path, "sssp.gem5"),
        "--options",
        f'{joinpath(dataset_path, "USA-road-d.NY.gr")} 0',
    ],
    valid_isas=(constants.vega_x86_tag,),
    valid_hosts=(constants.host_gcn_gpu_tag,),
    length=constants.very_long_tag,
)

# Docker image support matrix

AMD validates and publishes
[PyTorch](https://hub.docker.com/r/rocm/pytorch) and
[TensorFlow](https://hub.docker.com/r/rocm/tensorflow) containers on
Docker Hub. The following tag links, and associated inventories, are validated
with ROCm 5.7.

## PyTorch Ubuntu 22.04

| [rocm5.7/ubuntu22.04/pytorch_2.0.1] |
| ----------------------------------- |
| [ROCm 5.7 apt]                      |
| [Python 3.10]                       |
| [Torch 2.0.1]                       |
| [Apex 0.1]                          |
| [Torchvision 0.15.0]                |
| [Tensorboard 2.14.0]                |
| [MAGMA]                             |
| [UCX 1.10.0]                        |
| [OMPI 4.0.3]                        |
| [OFED 5.4.3]                        |

## PyTorch Ubuntu 20.04

| [rocm5.7/pytorch_staging] | [rocm5.7/pytorch_1.12.1] | [rocm5.7/pytorch_1.13.1] | [rocm5.7/pytorch_2.0.1] |
| ------------------------- | ------------------------ | ------------------------ | ----------------------- |
| [ROCm 5.7 apt]            | [ROCm 5.7 apt]           | [ROCm 5.7 apt]           | [ROCm 5.7 apt]          |
| [Python 3.9]              | [Python 3.9]             | [Python 3.9]             | [Python 3.9]            |
| [Torch 2.1.0]             | [Torch 1.12.1]           | [Torch 1.13.1]           | [Torch 2.0.1]           |
| [Apex 0.1]                | [Apex 0.1]               | [Apex 0.1]               | [Apex 0.1]              |
| [Torchvision 0.16.0]      | [Torchvision 0.13.1]     | [Torchvision 0.14.0]     | [Torchvision 0.15.2]    |
| [Tensorboard 2.14.0]      | [Tensorboard 2.14.0]     | [Tensorboard 2.12.0]     | [Tensorboard 2.14.0]    |
| [MAGMA]                   | [MAGMA]                  | [MAGMA]                  | [MAGMA]                 |
| [UCX 1.10.0]              | [UCX 1.10.0]             | [UCX 1.10.0]             | [UCX 1.10.0]            |
| [OMPI 4.0.3]              | [OMPI 4.0.3]             | [OMPI 4.0.3]             | [OMPI 4.0.3]            |
| [OFED 5.4.3]              | [OFED 5.4.3]             | [OFED 5.4.3]             | [OFED 5.4.3]            |

## PyTorch CentOS 7

| [rocm5.7/centos7/pytorch_staging] |
| --------------------------------- |
| [ROCm 5.7 yum]                    |
| [Python 3.9]                      |
| [Torch 2.1.0]                     |
| [Apex 0.1]                        |
| [Torchvision 0.16.0]              |
| [MAGMA]                           |

## TensorFlow Ubuntu 20.04

| [rocm5.7-tf2.12-dev]     | [rocm5.7-tf2.13-dev]     |
| ------------------------ | ------------------------ |
| [ROCm 5.7 apt]           | [ROCm 5.7 apt]           |
| [Python 3.9]             | [Python 3.9]             |
| [tensorflow-rocm 2.12.1] | [tensorflow-rocm 2.13.0] |
| [Tensorboard 2.12.3]     | [Tensorboard 2.13.0]     |

[Apex 0.1]: https://github.com/ROCmSoftwarePlatform/apex/tree/v0.1
[MAGMA]: https://bitbucket.org/icl/magma/src/master/
[OFED 5.4.3]: https://content.mellanox.com/ofed/MLNX_OFED-5.3-1.0.5.0/MLNX_OFED_LINUX-5.3-1.0.5.0-ubuntu20.04-x86_64.tgz
[OMPI 4.0.3]: https://github.com/open-mpi/ompi/tree/v4.0.3
[Python 3.10]: https://www.python.org/downloads/release/python-31013/
[Python 3.9]: https://www.python.org/downloads/release/python-3918/
[ROCm 5.7 apt]: https://repo.radeon.com/rocm/apt/5.7/
[ROCm 5.7 yum]: https://repo.radeon.com/rocm/yum/5.7/
[Tensorboard 2.12.0]: https://github.com/tensorflow/tensorboard/tree/2.12.0
[Tensorboard 2.12.3]: https://github.com/tensorflow/tensorboard/tree/2.12
[Tensorboard 2.13.0]: https://github.com/tensorflow/tensorboard/tree/2.13
[Tensorboard 2.14.0]: https://github.com/tensorflow/tensorboard/tree/2.14
[Torch 1.12.1]: https://github.com/ROCmSoftwarePlatform/pytorch/tree/release/1.12
[Torch 1.13.1]: https://github.com/ROCmSoftwarePlatform/pytorch/tree/release/1.13
[Torch 2.0.1]: https://github.com/ROCmSoftwarePlatform/pytorch/tree/release/2.0
[Torch 2.1.0]: https://github.com/ROCmSoftwarePlatform/pytorch/tree/rocm5.7_internal_testing
[Torchvision 0.13.1]: https://github.com/pytorch/vision/tree/v0.13.1
[Torchvision 0.14.0]: https://github.com/pytorch/vision/tree/v0.14.0
[Torchvision 0.15.0]: https://github.com/pytorch/vision/tree/release/0.15
[Torchvision 0.15.2]: https://github.com/pytorch/vision/tree/release/0.15
[Torchvision 0.16.0]: https://github.com/pytorch/vision/tree/release/0.16
[UCX 1.10.0]: https://github.com/openucx/ucx/tree/v1.10.0
[rocm5.7/centos7/pytorch_staging]: https://hub.docker.com/layers/rocm/pytorch/rocm5.7_centos7_py3.9_pytorch_staging/images/sha256-92240cdf0b4aa7afa76fc78be995caa19ee9c54b5c9f1683bdcac28cedb58d2b
[rocm5.7/pytorch_1.12.1]: https://hub.docker.com/layers/rocm/pytorch/rocm5.7_ubuntu20.04_py3.9_pytorch_1.12.1/images/sha256-e67db9373c045a7b6defd43cc3d067e7d49fd5d380f3f8582d2fb219c1756e1f
[rocm5.7/pytorch_1.13.1]: https://hub.docker.com/layers/rocm/pytorch/rocm5.7_ubuntu20.04_py3.9_pytorch_1.13.1/images/sha256-ed99d159026093d2aaf5c48c1e4b0911508773430377051372733f75c340a4c1
[rocm5.7/pytorch_2.0.1]: https://hub.docker.com/layers/rocm/pytorch/rocm5.7_ubuntu20.04_py3.9_pytorch_2.0.1/images/sha256-4dd86046e5f777f53ae40a75ecfc76a5e819f01f3b2d40eacbb2db95c2f971d4
[rocm5.7/pytorch_staging]: https://hub.docker.com/layers/rocm/pytorch/rocm5.7_ubuntu20.04_py3.9_pytorch_2.0.1/images/sha256-4dd86046e5f777f53ae40a75ecfc76a5e819f01f3b2d40eacbb2db95c2f971d4
[rocm5.7/ubuntu22.04/pytorch_2.0.1]: https://hub.docker.com/layers/rocm/pytorch/rocm5.7_ubuntu22.04_py3.10_pytorch_2.0.1/images/sha256-21df283b1712f3d73884b9bc4733919374344ceacb694e8fbc2c50bdd3e767ee
[rocm5.7-tf2.12-dev]: https://hub.docker.com/layers/rocm/tensorflow/rocm5.7-tf2.12-dev/images/sha256-e0ac4d49122702e5167175acaeb98a79b9500f585d5e74df18facf6b52ce3e59
[rocm5.7-tf2.13-dev]: https://hub.docker.com/layers/rocm/tensorflow/rocm5.7-tf2.13-dev/images/sha256-6f995539eebc062aac2b53db40e2b545192d8b032d0deada8c24c6651a7ac332
[tensorflow-rocm 2.12.1]: https://pypi.org/project/tensorflow-rocm/2.12.1.570/
[tensorflow-rocm 2.13.0]: https://pypi.org/project/tensorflow-rocm/2.13.0.570/

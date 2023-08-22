# Installing PyTorch for ROCm

[PyTorch](https://pytorch.org/) is an open-source tensor library designed for deep learning. PyTorch on
ROCm provides mixed-precision and large-scale training using our
[MIOpen](https://github.com/ROCmSoftwarePlatform/MIOpen) and
[RCCL](https://github.com/ROCmSoftwarePlatform/rccl) libraries.

To install [PyTorch for ROCm](https://pytorch.org/blog/pytorch-for-amd-rocm-platform-now-available-as-python-package/), you have the following options:

* [Use a Docker image with PyTorch pre-installed](#using-a-docker-image-with-pytorch-pre-installed)
  (recommended)
* [Use a wheels package](#using-a-wheels-package)
* [Use the PyTorch ROCm Base Docker Image](#using-the-pytorch-rocm-base-docker-image)
* [Use the PyTorch Upstream Docker File](#using-the-pytorch-upstream-docker-file)

For hardware, software, and third-party framework compatibility between ROCm and PyTorch, refer to:

* [GPU and OS Support (Linux)](../../release/gpu_os_support.md)
* [Compatibility](../../release/compatibility.md)

## Using a Docker image with PyTorch pre-installed

1. Download the latest public PyTorch Docker image
   ([https://hub.docker.com/r/rocm/pytorch](https://hub.docker.com/r/rocm/pytorch)).

   ```bash
   docker pull rocm/pytorch:latest
   ```

   You can also download a specific and supported configuration with different user-space ROCm
   versions, PyTorch versions, and operating systems.

2. Start a Docker container using the downloaded image.

   ```bash
   docker run -it --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 8G rocm/pytorch:latest
   ```

   :::{note}
   This will automatically download the image if it does not exist on the host. You can also pass the `-v`
   argument to mount any data directories from the host onto the container.
   :::

(install_pytorch_wheels)=

## Using a wheels package

PyTorch supports the ROCm platform by providing tested wheels packages. To access this feature, go
to [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/). In the interactive
table, choose ROCm from the _Compute Platform_ row.

1. Choose one of the following three options:

   **Option 1:**

   Download a base Docker image with the correct user-space ROCm version. See:
   [https://hub.docker.com/repository/docker/rocm/dev-ubuntu-20.04](https://hub.docker.com/repository/docker/rocm/dev-ubuntu-20.04).

   **Option 2:**

   Download a base OS Docker image and install ROCm using the directions in the
   [Installation](../../deploy/linux/install.md) section.

   **Option 3:**

   Install on bare metal. If using this method, skip step 2 (starting a Docker container).

   ```bash
   docker run -it --device=/dev/kfd --device=/dev/dri --group-add video rocm/dev-ubuntu-20.04:latest
   ```

2. Start the Docker container (skip this step for bare metal installations).

   ```dockerfile
   docker run -it --device=/dev/kfd --device=/dev/dri --group-add video rocm/dev-ubuntu-20.04:latest
   ```

3. Install the required dependencies for the wheels package.

   ```bash
   sudo apt update
   sudo apt install libjpeg-dev python3-dev
   pip3 install wheel setuptools
   ```

4. Install torch, `torchvision`, and `torchaudio`, as specified in the _Run this Command_ column of the
   interactive table on the [PyTorch Start Locally](https://pytorch.org/get-started/locally/) website.

   :::{note}
   The following command uses the ROCm 5.4.2 PyTorch wheel. If you want a different version of
   ROCm, modify the command accordingly.
   :::

   ```bash
   pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/rocm5.4.2/
   ```

## Using the PyTorch ROCm base Docker image

The pre-built base Docker image has all dependencies installed, including:

* ROCm
* Torchvision
* Conda packages
* The compiler toolchain

Additionally, a particular environment flag (`BUILD_ENVIRONMENT`) is set, which is used by the build
scripts to determine the configuration of the build environment.

1. Download the Docker image. This is the base container, which does not contain PyTorch.

   ```bash
   docker pull rocm/pytorch:latest-base
   ```

2. Start a Docker container using the downloaded image.

   ```bash
   docker run -it --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 8G rocm/pytorch:latest-base
   ```

   You can also pass the `-v` argument to mount any data directories from the host onto the container.

3. Clone the PyTorch repository.

   ```bash
   cd ~
   git clone https://github.com/pytorch/pytorch.git
   cd pytorch
   git submodule update --init --recursive
   ```

4. Build PyTorch for ROCm. The default build directory is `rocm/pytorch:latest-base`.

   :::{note}
   PyTorch builds simultaneously for the following architectures:
   * gfx900
   * gfx906
   * gfx908
   * gfx90a
   * gfx1030

   If you want to compile _only_ for your microarchitecture (uarch), run:

   ```bash
   export PYTORCH_ROCM_ARCH=<uarch>
   ```

   Where `<uarch>` is the architecture reported by the `rocminfo` command.

   To find your uarch, run:

   ```bash
   rocminfo | grep gfx
   ```
   :::

   You have two options for building PyTorch:

   **Option 1**

   ```bash
   ./.jenkins/pytorch/build.sh
   ```

   This converts PyTorch sources for
   [HIP compatibility](https://www.amd.com/en/developer/rocm-hub/hip-sdk.html) and builds the
   PyTorch framework.

   **Option 2**

   ```bash
   python3 tools/amd_build/build_amd.py
   USE_ROCM=1 MAX_JOBS=4 python3 setup.py install --user
   ```

## Using the PyTorch upstream Docker file

If you don't want to use a prebuilt base Docker image, you can build a custom base Docker image
using scripts from the PyTorch repository. This uses a standard Docker image from operating system
maintainers and installs all the required dependencies, including:

* ROCm
* Torchvision
* Conda packages
* The compiler toolchain

1. Clone the PyTorch repository.

   ```bash
   cd ~
   git clone https://github.com/pytorch/pytorch.git
   cd pytorch
   git submodule update --init --recursive
   ```

2. Build the PyTorch Docker image.

   ```bash
   cd .circleci/docker
   ./build.sh pytorch-linux-bionic-rocm<version>-py3.7
   # eg. ./build.sh pytorch-linux-bionic-rocm3.10-py3.7
   ```

   Once you see the message: "Successfully build `<image_id>`", your build is complete.

3. Start a Docker container using the image.

   ```bash
   docker run -it --cap-add=SYS_PTRACE --security-opt
   seccomp=unconfined --device=/dev/kfd --device=/dev/dri --group-add
   video --ipc=host --shm-size 8G <image_id>
   ```

   You can also pass the `-v` argument to mount any data directories from the host onto the container.

4. Clone the PyTorch repository.

   ```bash
   cd ~
   git clone https://github.com/pytorch/pytorch.git
   cd pytorch
   git submodule update --init --recursive
   ```

5. Build PyTorch for ROCm. The default build directory is `rocm/pytorch:latest-base`.

   :::{note}
   PyTorch builds simultaneously for the following architectures:
   * gfx900
   * gfx906
   * gfx908
   * gfx90a
   * gfx1030

   If you want to compile _only_ for your microarchitecture (uarch), run:

   ```bash
   export PYTORCH_ROCM_ARCH=<uarch>
   ```

   Where `<uarch>` is the architecture reported by the `rocminfo` command.

   To find your uarch, run:

   ```bash
   rocminfo | grep gfx
   ```
   :::

   You have two options for building PyTorch:

   **Option 1**

   ```bash
   ./.jenkins/pytorch/build.sh
   ```

   This converts PyTorch sources for
   [HIP compatibility](https://www.amd.com/en/developer/rocm-hub/hip-sdk.html) and builds the
   PyTorch framework.

   **Option 2**

   ```bash
   python3 tools/amd_build/build_amd.py
   USE_ROCM=1 MAX_JOBS=4 python3 setup.py install --user
   ```

## Test the PyTorch Installation

You can use PyTorch unit tests to validate your PyTorch installation. If you used a
**prebuilt PyTorch Docker image from AMD ROCm DockerHub** or installed an
**official wheels package**, validation tests are not necessary.

If you want to manually run unit tests to validate your PyTorch installation fully, follow these steps:

1. Import the torch package in Python to test if PyTorch is installed and accessible.

   :::{note}
   Do not run the following command in the PyTorch git folder.
   :::

   ```bash
   python3 -c 'import torch' 2> /dev/null && echo 'Success' || echo 'Failure'
   ```

2. Check if the GPU is accessible from PyTorch. In the PyTorch framework, `torch.cuda` is a generic way to
   access the GPU. This can only access an AMD GPU if one is available.

   ```bash
   python3 -c 'import torch; print(torch.cuda.is_available())'
   ```

3. Run unit tests to validate the PyTorch installation fully.

   :::{note}
   You must run the following command from the PyTorch home directory.
   :::

   ```bash
   BUILD_ENVIRONMENT=${BUILD_ENVIRONMENT:-rocm} ./.jenkins/pytorch/test.sh
   ```

   This command ensures that the required environment variable is set to skip certain unit tests for
   ROCm. This also applies to wheel installs in a non-controlled environment. During this step,
   dependencies (such as `torchvision`) are installed.

   :::{note}
   Make sure your PyTorch source code corresponds to the PyTorch wheel or the installation in the
   Docker image. Incompatible PyTorch source code can give errors when running unit tests.
   :::

   Some tests may be skipped, as appropriate, based on your system configuration. ROCm doesn't
   support all PyTorch features; tests that evaluate unsupported features are skipped. Other tests might
   be skipped, depending on the host memory and number of available GPUs.

   If the compilation and installation are correct, all tests will pass.

4. Run individual unit tests.

   ```bash
   PYTORCH_TEST_WITH_ROCM=1 python3 test/test_nn.py --verbose
   ```

   You can replace `test_nn.py` with any other test set.

## Run a Basic PyTorch Example

The PyTorch examples repository provides basic examples that exercise the functionality of your
framework.

Two of our favorite testing databases are:

* **MNIST** (Modified National Institute of Standards and Technology): A database of handwritten
  digits that can be used to train a Convolutional Neural Network for **handwriting recognition**.
* **ImageNet**: A database of images that can be used to train a network for
  **visual object recognition**.

### MNIST PyTorch example

1. Clone the PyTorch examples repository.

   ```bash
   git clone https://github.com/pytorch/examples.git
   ```

2. Run the MNIST example.

   ```bash
   cd examples/mnist
   ```

3. Follow the instructions in the `README` file.

   ```bash
   pip3 install -r requirements.txt
   python3 main.py
   ```

### ImageNet PyTorch example

1. Clone the PyTorch examples repository.

   ```bash
   git clone https://github.com/pytorch/examples.git
   ```

2. Run the ImageNet example.

   ```bash
   cd examples/imagenet
   ```

3. Follow the instructions in the `README` file.

   ```bash
   pip3 install -r requirements.txt
   python3 main.py
   ```

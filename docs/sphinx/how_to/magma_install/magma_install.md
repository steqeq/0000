# Magma Installation for ROCm

Pull content from
<https://docs.amd.com/bundle/ROCm-Deep-Learning-Guide-v5.4.1/page/Frameworks_Installation.html>

The following sections cover the different framework installations for ROCm and
Deep Learning applications. Figure 5 provides the sequential flow for the use of
each framework. Refer to the ROCm Compatible Frameworks Release Notes for each
framework's most current release notes at
[/bundle/ROCm-Compatible-Frameworks-Release-Notes/page/Framework_Release_Notes.html](/bundle/ROCm-Compatible-Frameworks-Release-Notes/page/Framework_Release_Notes.html).

| ![Figure 5](figures/image.005.png)|
|:--:|
| <b>Figure 5. ROCm Compatible Frameworks Flowchart</b>|

## PyTorch
PyTorch is an open source Machine Learning Python library, primarily differentiated by Tensor computing with GPU acceleration and a type-based automatic differentiation. Other advanced features include:
- Support for distributed training
- Native ONNX support
- C++ frontend
- The ability to deploy at scale using TorchServe
- A production-ready deployment mechanism through TorchScript
### Installing PyTorch
To install ROCm on bare metal, refer to the section [ROCm Installation](https://docs.amd.com/bundle/ROCm-Deep-Learning-Guide-v5.4-/page/Prerequisites.html#d2999e60). The recommended option to get a PyTorch environment is through Docker. However, installing the PyTorch wheels package on bare metal is also supported.
#### Option 1 (Recommended): Use Docker Image with PyTorch Pre-installed
Using Docker gives you portability and access to a prebuilt Docker container that has been rigorously tested within AMD. This might also save on the compilation time and should perform as it did when tested without facing potential installation issues.
Follow these steps:
1. Pull the latest public PyTorch Docker image.

```
docker pull rocm/pytorch:latest
```

Optionally, you may download a specific and supported configuration with different user-space ROCm versions, PyTorch versions, and supported operating systems. To download the PyTorch Docker image, refer to [https://hub.docker.com/r/rocm/pytorch](https://hub.docker.com/r/rocm/pytorch).

2. Start a Docker container using the downloaded image.

```
docker run -it --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 8G rocm/pytorch:latest
```

:::{note}
This will automatically download the image if it does not exist on the host. You can also pass the -v argument to mount any data directories from the host onto the container.
:::

#### Option 2: Install PyTorch Using Wheels Package
PyTorch supports the ROCm platform by providing tested wheels packages. To access this feature, refer to [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) and choose the "ROCm" compute platform. Figure 6 is a matrix from pytroch.org that illustrates the installation compatibility between ROCm and the PyTorch build.


| ![Figure 6](figures/image.006.png)|
|:--:|
| <b>Figure 6.  Installation Matrix from Pytorch.org</b>|


To install PyTorch using the wheels package, follow these installation steps:

1. Choose one of the following options:

a. Obtain a base Docker image with the correct user-space ROCm version installed from [https://hub.docker.com/repository/docker/rocm/dev-ubuntu-20.04](https://hub.docker.com/repository/docker/rocm/dev-ubuntu-20.04).

 or

b.  Download a base OS Docker image and install ROCm following the installation directions in the section [Installation](https://docs.amd.com/bundle/ROCm-Deep-Learning-Guide-v5.4-/page/Prerequisites.html#d2999e60). ROCm 5.2 is installed in this example, as supported by the installation matrix from pytorch.org.

or

 c.  Install on bare metal. Skip to Step 3.

```
docker run -it --device=/dev/kfd --device=/dev/dri --group-add video rocm/dev-ubuntu-20.04:latest
```
3. Install any dependencies needed for installing the wheels package.

```
sudo apt update
sudo apt install libjpeg-dev python3-dev
pip3 install wheel setuptools
```

4. Install torch, torchvision, and torchaudio as specified by the installation matrix.

:::{note}
ROCm 5.2 PyTorch wheel in the command below is shown for reference.
:::


```
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/rocm5.2/
```

#### Option 3: Install PyTorch Using PyTorch ROCm Base Docker Image
A prebuilt base Docker image is used to build PyTorch in this option. The base Docker has all dependencies installed, including:
- ROCm
- Torchvision
- Conda packages
- Compiler toolchain
Additionally, a particular environment flag (BUILD_ENVIRONMENT) is set, and the build scripts utilize that to determine the build environment configuration.

Follow these steps:

1. Obtain the Docker image.
```
docker pull rocm/pytorch:latest-base
```
The above will download the base container, which does not contain PyTorch.

2. Start a Docker container using the image.
```
docker run -it --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 8G rocm/pytorch:latest-base
```
You can also pass the -v argument to mount any data directories from the host onto the container.

3. Clone the PyTorch repository.
```
cd ~
git clone https://github.com/pytorch/pytorch.git
cd pytorch
git submodule update --init –recursive
```

4. Build PyTorch for ROCm.
:::{note}
By default in the rocm/pytorch:latest-base, PyTorch builds for these architectures simultaneously:
- gfx900
- gfx906
- gfx908
- gfx90a
- gfx1030
:::

5. To determine your AMD uarch, run:
```
rocminfo | grep gfx
```

6. In the event you want to compile only for your uarch, use:
```
export PYTORCH_ROCM_ARCH=<uarch>
```
\<uarch\> is the architecture reported by the rocminfo command. is the architecture reported by the rocminfo command.

7.  Build PyTorch using the following command:
```
./.jenkins/pytorch/build.sh
```
This will first convert PyTorch sources for HIP compatibility and build the PyTorch framework.

8. Alternatively, build PyTorch by issuing the following commands:
```
python3 tools/amd_build/build_amd.py
USE_ROCM=1 MAX_JOBS=4 python3 setup.py install ––user
```
#### Option 4: Install Using PyTorch Upstream Docker File
Instead of using a prebuilt base Docker image, you can build a custom base Docker image using scripts from the PyTorch repository. This will utilize a standard Docker image from operating system maintainers and install all the dependencies required to build PyTorch, including
- ROCm
- Torchvision
- Conda packages
- Compiler toolchain

Follow these steps:

1. Clone the PyTorch repository on the host.
```
cd ~
git clone https://github.com/pytorch/pytorch.git
cd pytorch
git submodule update --init –recursive
```

2. Build the PyTorch Docker image.
```
cd.circleci/docker
./build.sh pytorch-linux-bionic-rocm<version>-py3.7 
# eg. ./build.sh pytorch-linux-bionic-rocm3.10-py3.7
```
This should be complete with a message "Successfully build &lt;image_id&gt;."

3. Start a Docker container using the image:
```
docker run -it --cap-add=SYS_PTRACE --security-opt 
seccomp=unconfined --device=/dev/kfd --device=/dev/dri --group-add 
video --ipc=host --shm-size 8G <image_id>
```
You can also pass -v argument to mount any data directories from the host onto the container.

4. Clone the PyTorch repository.
```
cd ~
git clone https://github.com/pytorch/pytorch.git
cd pytorch
git submodule update --init --recursive
```

5. Build PyTorch for ROCm.
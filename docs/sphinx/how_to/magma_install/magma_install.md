# Magma Installation for ROCm

Pull content from
<https://docs.amd.com/bundle/ROCm-Deep-Learning-Guide-v5.4.1/page/Frameworks_Installation.html>

The following sections cover the different framework installations for ROCm and
Deep Learning applications. Figure 5 provides the sequential flow for the use of
each framework. Refer to the ROCm Compatible Frameworks Release Notes for each
framework's most current release notes at
/bundle/ROCm-Compatible-Frameworks-Release-Notes/page/Framework_Release_Notes.html.
![Figure 5](figures/image.005.png)
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
---
**NOTE**

This will automatically download the image if it does not exist on the host. You can also pass the -v argument to mount any data directories from the host onto the container.

---
#### Option 2: Install PyTorch Using Wheels Package
PyTorch supports the ROCm platform by providing tested wheels packages. To access this feature, refer to [https://pytorch.org/get-started/locally/] (https://pytorch.org/get-started/locally/) and choose the "ROCm" compute platform. Figure 6 is a matrix from [pytroch.org] (pytroch.org) that illustrates the installation compatibility between ROCm and the PyTorch build.


<head>
  <meta charset="UTF-8">
  <meta name="description" content="AMD ROCm documentation">
  <meta name="keywords" content="documentation, guides, installation, compatibility, support,
  reference">
</head>

# AMD ROCm™ documentation

Welcome to the ROCm docs home page! If you're new to ROCm, you can review the following
resources to learn more about our products and what we support:

* [What is ROCm?](./what-is-rocm.md)
* [What's new?](about/whats-new/whats-new)
* [Release notes](./about/release-notes.md)

Our documentation is organized into the following categories:

::::{grid} 1 2 2 2
:class-container: rocm-doc-grid

:::{grid-item-card}
:padding: 2
**Installation**

Installation guides
^^^

* Linux
  * {doc}`Quick-start <linux-install-docs:tutorial/quick-start>`
  * {doc}`Linux install guide <linux-install-docs:tutorial/install-overview>`
  * {doc}`Package manager integration <linux-install-docs:how-to/amdgpu-install>`

* Windows
  * [Quick-start (Windows)](./install/windows/install-quick.md)
  * [Windows install guide](./install/windows/install.md)
  * [Application deployment guidelines](./install/windows/windows-app-deployment-guidelines.md)
* {doc}`ROCm & Docker containers <linux-install-docs:how-to/docker>`
* {doc}`PyTorch for ROCm <linux-install-docs:how-to/3rd-party/pytorch-install>`
* {doc}`TensorFlow for ROCm <linux-install-docs:how-to/3rd-party/tensorflow-install>`
* {doc}`MAGMA for ROCm <linux-install-docs:how-to/3rd-party/magma-install>`
* {doc}`ROCm & Spack <linux-install-docs:how-to/spack>`

:::

:::{grid-item-card}
:padding: 2
**Compatibility & Support**

ROCm compatibility information
^^^

* [Linux (GPU & OS)](./about/compatibility/linux-support.md)
* [Windows (GPU & OS)](./about/compatibility/windows-support.md)
* {doc}`Third-party <linux-install-docs:reference/3rd-party-support-matrix>`
* {doc}`User/kernel space <linux-install-docs:reference/user-kernel-space-compat-matrix>`
* {doc}`Docker <linux-install-docs:reference/docker-image-support-matrix>`

:::

:::{grid-item-card}
:padding: 2
**How-to**

Task-oriented walkthroughs
^^^

* [System tuning for various architectures](./how-to/tuning-guides.md)
  * [MI100](./how-to/tuning-guides/mi100.md)
  * [MI200](./how-to/tuning-guides/mi200.md)
  * [RDNA2](./how-to/tuning-guides/w6000-v620.md)
* [Setting up for deep learning with ROCm](./how-to/deep-learning-rocm.md)
* [GPU-enabled MPI](./how-to/gpu-enabled-mpi.rst)
* [System level debugging](./how-to/system-debugging.md)
* [GitHub examples](https://github.com/amd/rocm-examples)

:::

:::{grid-item-card}
:padding: 2
**Reference**

Collated information
^^^

* [API Libraries](./reference/library-index.md)

:::

:::{grid-item-card}
:padding: 2
**Conceptual**

Topic overviews & background information
^^^

* [GPU architecture](./conceptual/gpu-arch.md)
  * [MI100](./conceptual/gpu-arch/mi100.md)
  * [MI200](./conceptual/gpu-arch/mi200-performance-counters.md)
  * [MI250](./conceptual/gpu-arch/mi250.md)
* [GPU memory](./conceptual/gpu-memory.md)
* [Compiler disambiguation](./conceptual/compiler-disambiguation.md)
* [File structure (Linux FHS)](./conceptual/file-reorg.md)
* [GPU isolation techniques](./conceptual/gpu-isolation.md)
* [LLVN ASan](./conceptual/using-gpu-sanitizer.md)
* [Using CMake](./conceptual/cmake-packages.rst)
* [ROCm & PCIe atomics](./conceptual/More-about-how-ROCm-uses-PCIe-Atomics.rst)
* [Inception v3 with PyTorch](./conceptual/ai-pytorch-inception.md)
* [Inference optimization with MIGraphX](./conceptual/ai-migraphx-optimization.md)
* [OpenMP support in ROCm](./conceptual/openmp.md)

:::

::::

We welcome collaboration! If you'd like to contribute to our documentation, you can find instructions
on our [Contribute to ROCm docs](./contribute/index.md) page. Known issues are listed on
[GitHub](https://github.com/RadeonOpenCompute/ROCm/labels/Verified%20Issue).

Licensing information for all ROCm components is listed on our [Licensing](./about/license.md) page.

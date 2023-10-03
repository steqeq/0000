# AMD ROCm™ documentation

Welcome to the ROCm docs home page! If you're new to ROCm, you can review the following
resources to learn more about our products and what we support:

* [What is ROCm?](./what-is-rocm.md)
* [What's new?](about/whats-new/whats-new)
* [Compatibility & support](./about/compatibility/index.md)
* [Release notes](./about/release-notes.md)

::::{grid} 1 2 2 2
:class-container: rocm-doc-grid

:::{grid-item-card}
:padding: 2
**Installation**

Installation guides
^^^

* [Linux quick-start](./install/linux/install-quick.md)
* [Windows quick-start](./install/windows/install-quick.md)
* [ROCm on Linux](./install/linux/install-quick.md)
* [ROCm on Windows](./install/windows/install-quick.md)
* [MAGMA for ROCm](./install/magma-install.md)
* [PyTorch for ROCm](./install/pytorch-install.md)
* [TensorFlow for ROCm](./install/tensorflow-install.md)

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
* [GPU-enabled MPI](./how-to/gpu-enabled-mpi.md)
* [System level debugging](./how-to/system-debugging.md)
* [ROCm & Spack](./how-to/spack.md)
* [GitHub examples](https://github.com/amd/rocm-examples)

:::

:::{grid-item-card}
:padding: 2
**Reference**

Collated information
^^^

* [Libraries](./reference/libraries/index.md)
  * [Math libraries](./reference/libraries/gpu-libraries/math.md)
  * [C++ primitives libraries](./reference/libraries/gpu-libraries/c++primitives.md)
  * [Communication libraries](./reference/libraries/gpu-libraries/communication.md)
* [Compilers & tools](./reference/compilers-tools/index.md)
  * [Management tools](./reference/compilers-tools/management-tools.md)
  * [Validation tools](./reference/compilers-tools/validation-tools.md)
* [HIP](./reference/hip.md)
* [OpenMP](./reference/openmp/openmp.md)

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

:::

::::

We welcome collaboration! If you'd like to contribute to our documentation, you can find instructions
on our [Contribute to ROCm docs](./contribute/index.md) page. Known issues are listed on
[GitHub](https://github.com/RadeonOpenCompute/ROCm/labels/Verified%20Issue).

Licensing information for all ROCm components is listed on our [Licensing](./about/license.md) page.

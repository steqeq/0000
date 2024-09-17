<head>
  <meta charset="UTF-8">
  <meta name="description" content="AMD ROCm documentation">
  <meta name="keywords" content="documentation, guides, installation, compatibility, support,
  reference, ROCm, AMD">
</head>

# AMD ROCm documentation

ROCm is an open-source software platform optimized to extract HPC and AI workload
performance from AMD Instinct accelerators and AMD Radeon GPUs while maintaining
compatibility with industry software frameworks. For more information, see [What is ROCm?](./what-is-rocm.rst)

If you're using Radeon GPUs, consider reviewing {doc}`Radeon-specific ROCm documentation<radeon:index>`.

Installation instructions are available from:

* {doc}`ROCm installation for Linux<rocm-install-on-linux:index>`
* {doc}`HIP SDK installation for Windows<rocm-install-on-windows:index>`
* [Deep learning frameworks installation](./how-to/deep-learning-rocm.rst)
* [Build ROCm from source](./how-to/build-rocm.rst)

ROCm documentation is organized into the following categories:

::::{grid} 1 2 2 2
:class-container: rocm-doc-grid

:::{grid-item-card}
:class-card: sd-text-black
:img-top: ./data/banner-compatibility.jpg
:img-alt: Compatibility information
:padding: 2

* [Compatibility matrix](./compatibility/compatibility-matrix.rst)
* {doc}`Linux system requirements<rocm-install-on-linux:reference/system-requirements>`
* {doc}`Windows system requirements<rocm-install-on-windows:reference/system-requirements>`
* {doc}`Third-party support<rocm-install-on-linux:reference/3rd-party-support-matrix>`
* {doc}`User/kernel space<rocm-install-on-linux:reference/user-kernel-space-compat-matrix>`
* {doc}`Docker<rocm-install-on-linux:reference/docker-image-support-matrix>`
* {doc}`OpenMP<llvm-project:conceptual/openmp>`
* [Precision support](./compatibility/precision-support.rst)
* {doc}`ROCm on Radeon GPUs<radeon:index>`
:::

:::{grid-item-card}
:class-card: sd-text-black
:img-top: ./data/banner-howto.jpg
:img-alt: How-to documentation
:padding: 2

* [Using ROCm for AI](./how-to/rocm-for-ai/index.rst)
* [Using ROCm for HPC](./how-to/rocm-for-hpc/index.rst)
* [Fine-tuning LLMs and inference optimization](./how-to/llm-fine-tuning-optimization/index.rst)
* [System optimization](./how-to/system-optimization/index.rst)
  * [AMD Instinct MI300X](./how-to/system-optimization/mi300x.rst)
  * [AMD Instinct MI300A](./how-to/system-optimization/mi300a.rst)
  * [AMD Instinct MI200](./how-to/system-optimization/mi200.md)
  * [AMD Instinct MI100](./how-to/system-optimization/mi100.md)
  * [AMD Instinct RDNA2](./how-to/system-optimization/w6000-v620.md)
* [AMD Instinct MI300X tuning guides](./how-to/tuning-guides/mi300x/index.rst)
  * [System tuning](./how-to/tuning-guides/mi300x/system.rst)
  * [Workload tuning](./how-to/tuning-guides/mi300x/workload.rst)
* [System debugging](./how-to/system-debugging.md)
* [GPU-enabled MPI](./how-to/gpu-enabled-mpi.rst)
* [Using advanced compiler features](./conceptual/compiler-topics.md)
* [Setting the number of CUs](./how-to/setting-cus)  
* [GitHub examples](https://github.com/amd/rocm-examples)
:::

:::{grid-item-card}
:class-card: sd-text-black
:img-top: ./data/banner-conceptual.jpg
:img-alt: Conceptual documentation
:padding: 2

* [GPU architecture](./conceptual/gpu-arch.md)
* [GPU memory](./conceptual/gpu-memory.md)
* [File structure (Linux FHS)](./conceptual/file-reorg.md)
* [GPU isolation techniques](./conceptual/gpu-isolation.md)
* [Using CMake](./conceptual/cmake-packages.rst)
* [ROCm & PCIe atomics](./conceptual/More-about-how-ROCm-uses-PCIe-Atomics.rst)
* [Inception v3 with PyTorch](./conceptual/ai-pytorch-inception.md)
* [Inference optimization with MIGraphX](./conceptual/ai-migraphx-optimization.md)
:::

<!-- markdownlint-disable MD051 -->
:::{grid-item-card}
:class-card: sd-text-black
:img-top: ./data/banner-reference.jpg
:img-alt: Reference documentation
:padding: 2

* [Libraries](./reference/api-libraries.md)
  * [Artificial intelligence](#artificial-intelligence-apis)
  * [C++ primitives](#cpp-primitives)
  * [Communication](#communication-libraries)
  * [Math](#math-apis)
  * [Random number generators](#random-number-apis)
  * [HIP runtime](#hip-runtime)
* [ROCm tools and compilers](./reference/rocm-tools.md)
* [GPU hardware specifications](./reference/gpu-arch-specs.rst)
:::
<!-- markdownlint-enable MD051 -->

::::

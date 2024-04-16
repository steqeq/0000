# ROCm 6.1 release highlights
<!-- Disable lints since this is an auto-generated file.    -->
<!-- markdownlint-disable blanks-around-headers             -->
<!-- markdownlint-disable no-duplicate-header               -->
<!-- markdownlint-disable no-blanks-blockquote              -->
<!-- markdownlint-disable ul-indent                         -->
<!-- markdownlint-disable no-trailing-spaces                -->

<!-- spellcheck-disable -->

The ROCm™ 6.1 release consists of new features and fixes to improve the stability and
performance of AMD Instinct™ MI300 GPU applications. Notably, we've added:

* Full support for Ubuntu 22.04.4.

* **rocDecode**, a new ROCm component that provides high-performance video decode support for
  AMD GPUs. With rocDecode, you can decode compressed video streams while keeping the resulting
  YUV frames in video memory. With decoded frames in video memory, you can run video
  post-processing using ROCm HIP, avoiding unnecessary data copies via the PCIe bus.

  To learn more, refer to the rocDecode 
  [documentation](https://rocm.docs.amd.com/projects/rocDecode/en/latest/).

## OS and GPU support changes

ROCm 6.1 adds the following operating system support:

* MI300A: Ubuntu 22.04.4 and RHEL 9.3
* MI300X: Ubuntu 22.04.4

Future releases will add additional operating systems to match the general offering. For older
generations of supported AMD Instinct products, we’ve added Ubuntu 22.04.4 support.

```{tip}
To view the complete list of supported GPUs and operating systems, refer to the system requirements
page for
[Linux](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)
and
[Windows](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html).
```

## Installation packages

This release includes a new set of packages for every module (all libraries and binaries default to
`DT_RPATH`). Package names have the suffix `rpath`; for example, the `rpath` variant of `rocminfo` is
`rocminfo-rpath`.

```{warning}
The new `rpath` packages will conflict with the default packages; they are meant to be used only in
environments where legacy `DT_RPATH` is the preferred form of linking (instead of `DT_RUNPATH`). We
do **not** recommend installing both sets of packages.
```

## ROCm components

The following sections highlight select component-specific changes. For additional details, refer to the
[Changelog](https://rocm.docs.amd.com/en/develop/about/CHANGELOG.html).

### AMD System Management Interface (SMI) Tool

* **New monitor command for GPU metrics**.
  Use the monitor command to customize, capture, collect, and observe GPU metrics on
  target devices.

* **Integration with E-SMI**.
  The EPYC™ System Management Interface In-band Library is a Linux C-library that provides in-band
  user space software APIs to monitor and control your CPU’s power, energy, performance, and other
  system management functionality. This integration enables access to CPU metrics and telemetry
  through the AMD SMI API and CLI tools.

### Composable Kernel (CK)

* **New architecture support**.
  CK now supports to the following architectures to enable efficient image denoising on the following
  AMD GPUs: gfx1030, gfx1100, gfx1031, gfx1101, gfx1032, gfx1102, gfx1034, gfx1103, gfx1035,
  gfx1036

* **FP8 rounding logic is replaced with stochastic rounding**.
  Stochastic rounding mimics a more realistic data behavior and improves model convergence.

### HIP

* **New environment variable to enable kernel run serialization**.
  The default `HIP_LAUNCH_BLOCKING` value is `0` (disable); which causes kernels to run as defined in
  the queue. When set to `1` (enable), the HIP runtime serializes the kernel queue, which behaves the
  same as `AMD_SERIALIZE_KERNEL`.

### hipBLASLt

* **New GemmTuning extension parameter** GemmTuning allows you to set a split-k value for each solution, which is more feasible for
  performance tuning.

### hipFFT

* **New multi-GPU support for single-process transforms** Multiple GPUs can be used to perform a transform in a single process. Note that this initial
  implementation is a functional preview.

### HIPIFY

* **Skipped code blocks**: Code blocks that are skipped by the preprocessor are no longer hipified under the
  `--default-preprocessor` option. To hipify everything, despite conditional preprocessor directives
  (`#if`, `#ifdef`, `#ifndef`, `#elif`, or `#else`), don't use the `--default-preprocessor` or `--amap` options.

### hipSPARSELt

* **Structured sparsity matrix support extensions**
  Structured sparsity matrices help speed up deep-learning workloads. We now support `B` as the
  sparse matrix and `A` as the dense matrix in Sparse Matrix-Matrix Multiplication (SPMM). Prior to this
  release, we only supported sparse (matrix A) x dense (matrix B) matrix multiplication. Structured
  sparsity matrices help speed up deep learning workloads.

### hipTensor

* **4D tensor permutation and contraction support**.
  You can now perform tensor permutation on 4D tensors and 4D contractions for F16, BF16, and
  Complex F32/F64 datatypes.

### MIGraphX

* **Improved performance for transformer-based models**.
  We added support for FlashAttention, which benefits models like BERT, GPT, and Stable Diffusion.

* **New Torch-MIGraphX driver**.
  This driver calls MIGraphX directly from PyTorch. It provides an `mgx_module` object that you can
  invoke like any other Torch module, but which utilizes the MIGraphX inference engine internally.
  Torch-MIGraphX supports FP32, FP16, and INT8 datatypes.

  * **FP8 support**. We now offer functional support for inference in the FP8E4M3FNUZ datatype. You
  can load an ONNX model in FP8E4M3FNUZ using C++ or Python APIs, or `migraphx-driver`.
  You can quantize a floating point model to FP8 format by using the `--fp8` flag with `migraphx-driver`.
  To accelerate inference, MIGraphX uses hardware acceleration on MI300 for FP8 by leveraging FP8
  support in various backend kernel libraries.

### MIOpen

* **Improved performance for inference and convolutions**.
  Inference support now provided for Find 2.0 fusion plans. Additionally, we've enhanced the Number of
  samples, Height, Width, and Channels (NHWC) convolution kernels for heuristics. NHWC stores data
  in a format where the height and width dimensions come first, followed by channels.

### OpenMP

* **Implicit Zero-copy is triggered automatically in XNACK-enabled MI300A systems**.
  Implicit Zero-copy behavior in `non unified_shared_memory` programs is triggered automatically in
  XNACK-enabled MI300A systems (for example, when using the `HSA_XNACK=1` environment
  variable). OpenMP supports the 'requires `unified_shared_memory`' directive to support programs
  that don’t want to copy data explicitly between the CPU and GPU. However, this requires that you add
  these directives to every translation unit of the program.

* **New MI300 FP atomics**. Application performance can now improve by leveraging fast floating-point atomics on MI300 (gfx942).
  

### RCCL

* **NCCL 2.18.6 compatibility**.
  RCCL is now compatible with NCCL 2.18.6, which includes increasing the maximum IB network interfaces to 32 and fixing network device ordering when creating communicators with only one GPU
  per node.

* **Doubled simultaneous communication channels**.
  We improved MI300X performance by increasing the maximum number of simultaneous
  communication channels from 32 to 64.

### rocALUTION

* **New multiple node and GPU support**.
  Unsmoothed and smoothed aggregations and Ruge-Stueben AMG now work with multiple nodes
  and GPUs. For more information, refer to the 
  [API documentation](https://rocm.docs.amd.com/projects/rocALUTION/en/latest/usermanual/solvers.html#unsmoothed-aggregation-amg).

### rocDecode

* **New ROCm component**.
  rocDecode ROCm's newest component, providing high-performance video decode support for AMD
  GPUs. To learn more, refer to the 
  [documentation](https://rocm.docs.amd.com/projects/rocDecode/en/latest/).

### ROCm Compiler

* **Combined projects**. ROCm Device-Libs, ROCm Compiler Support, and hipCC are now located in
  the `llvm-project/amd` subdirectory of AMD's fork of the LLVM project. Previously, these projects
  were maintained in separate repositories. Note that the projects themselves will continue to be
  packaged separately.

* **Split the 'rocm-llvm' package**. This package has been split into a required and an optional package: 

  * **rocm-llvm(required)**: A package containing the essential binaries needed for compilation.
  
  * **rocm-llvm-dev(optional)**: A package containing binaries for compiler and application developers.
    

### ROCm Data Center Tool (RDC)

* **C++ upgrades**.
  RDC was upgraded from C++11 to C++17 to enable a more modern C++ standard when writing RDC plugins.

### ROCm Performance Primitives (RPP)

* **New backend support**.
  Audio processing support added for the `HOST` backend and 3D Voxel kernels support
  for the `HOST` and `HIP` backends.

### ROCm Validation Suite

* **New datatype support**.
Added BF16 and FP8 datatypes based on General Matrix Multiply(GEMM) operations in the GPU Stress Test (GST) module. This provides additional performance benchmarking and stress testing based on the newly supported datatypes.

### rocSOLVER

* **New EigenSolver routine**.
Based on the Jacobi algorithm, a new EigenSolver routine was added to the library. This routine computes the eigenvalues and eigenvectors of a matrix with improved performance.

### ROCTracer

* **New versioning and callback enhancements**.
Improved to match versioning changes in HIP Runtime and supports runtime API callbacks and activity record logging. The APIs of different runtimes at different levels are considered different API domains with assigned domain IDs.

## Upcoming changes

* ROCm SMI will be deprecated in a future release. We advise **migrating to AMD SMI** now to
  prevent future workflow disruptions.

* hipCC supports, by default, the following compiler invocation flags:

  * `-mllvm -amdgpu-early-inline-all=true`
  * `-mllvm -amdgpu-function-calls=false`

  In a future ROCm release, hipCC will no longer support these flags. It will, instead, use the Clang
  defaults:

  * `-mllvm -amdgpu-early-inline-all=false`
  * `-mllvm -amdgpu-function-calls=true`

  To evaluate the impact of this change, include `--hipcc-func-supp` in your hipCC invocation.

  For information on these flags, and the differences between hipCC and Clang, refer to
  [ROCm Compiler Interfaces](https://rocm.docs.amd.com/en/latest/reference/rocmcc.html#rocm-compiler-interfaces).

*  Future ROCm releases will not provide `clang-ocl`. For more information, refer to the
  [`clang-ocl` README](https://github.com/ROCm/clang-ocl).

* The following operating systems will be supported in a future ROCm release. They are currently
  only available in beta.

  * RHEL 9.4
  * RHEL 8.10
  * SLES 15 SP6

* As of ROCm 6.2, we’ve planned for **end-of-support** for:

  * Ubuntu 20.04.5
  * SLES 15 SP4
  * RHEL/CentOS 7.9

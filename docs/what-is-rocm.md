<head>
  <meta charset="UTF-8">
  <meta name="description" content="What is ROCm">
  <meta name="keywords" content="documentation, projects, introduction, ROCm, AMD">
</head>

# What is ROCm?

ROCm is an open-source stack, composed primarily of open-source software, designed for
graphics processing unit (GPU) computation. ROCm consists of a collection of drivers, development
tools, and APIs that enable GPU programming from low-level kernel to end-user applications.

With ROCm, you can customize your GPU software to meet your specific needs. You can develop,
collaborate, test, and deploy your applications in a free, open source, integrated, and secure software
ecosystem. ROCm is particularly well-suited to GPU-accelerated high-performance computing (HPC),
artificial intelligence (AI), scientific computing, and computer aided design (CAD).

ROCm is powered by AMD’s
[Heterogeneous-computing Interface for Portability (HIP)](https://rocm.docs.amd.com/projects/HIP/en/latest/index.html),
an open-source software C++ GPU programming environment and its corresponding runtime. HIP
allows ROCm developers to create portable applications on different platforms by deploying code on a
range of platforms, from dedicated gaming GPUs to exascale HPC clusters.

ROCm supports programming models, such as OpenMP and OpenCL, and includes all necessary open
source software compilers, debuggers, and libraries. ROCm is fully integrated into machine learning
(ML) frameworks, such as PyTorch and TensorFlow.

```{tip}
  If you're using Radeon GPUs, refer to the
{doc}`Radeon-specific ROCm documentation<radeon:index>`
```

## ROCm projects

ROCm consists of the following drivers, development tools, and APIs.

| Project | Description |
| :---------------- | :------------ |
| [AMD Compute Language Runtimes (CLR)](https://github.com/ROCm/clr) | Contains source code for AMD's compute languages runtimes: {doc}`HIP <hip:index>` and OpenCL |
| {doc}`AMD SMI <amdsmi:index>` | A C library for Linux that provides a user space interface for applications to monitor and control AMD devices |
| [AOMP](https://github.com/ROCm/aomp/) | A scripted build of [LLVM](https://github.com/ROCm/llvm-project) and supporting software |
| [Asynchronous Task and Memory Interface (ATMI)](https://github.com/ROCm/atmi/) | A runtime framework for efficient task management in heterogeneous CPU-GPU systems |
| {doc}`Composable Kernel <composable_kernel:index>` | A library that aims to provide a programming model for writing performance critical kernels for machine learning workloads across multiple architectures |
| [Flang](https://github.com/ROCm/flang/) | An out-of-tree Fortran compiler targeting LLVM |
| [Half-precision floating point library (half)](https://github.com/ROCm/half/) | A C++ header-only library that provides an IEEE 754 conformant, 16-bit half-precision floating-point type along with corresponding arithmetic operators, type conversions, and common mathematical functions |
| {doc}`HIP <hip:index>` | AMD’s GPU programming language extension and the GPU runtime |
| {doc}`hipBLAS <hipblas:index>` | A BLAS-marshaling library that supports [rocBLAS](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/) and cuBLAS backends |
| {doc}`hipBLASLt <hipblaslt:index>` | A library that provides general matrix-matrix operations with a flexible API and extends functionalities beyond traditional BLAS library |
| [hipCC](https://github.com/ROCm/HIPCC) | A compiler driver utility that calls Clang or NVCC and passes the appropriate include and library options for the target compiler and HIP infrastructure |
| {doc}`hipCUB <hipcub:index>` | A thin header-only wrapper library on top of [rocPRIM](https://rocm.docs.amd.com/projects/rocPRIM/en/latest/) or CUB that allows project porting using the CUB library to the HIP layer |
| {doc}`hipFFT <hipfft:index>` | A fast Fourier transforms (FFT)-marshalling library that supports rocFFT or cuFFT backends |
| {doc}`hipfort <hipfort:index>` | A Fortran interface library for accessing GPU Kernels |
| {doc}`HIPIFY <hipify:index>` | A set of tools for translating CUDA source code into portable HIP C++ |
| {doc}`hipRAND <hiprand:index>` | A wrapper library to easily port CUDA applications that use the cuRAND library into the HIP layer |
| {doc}`hipSOLVER <hipsolver:index>` | An LAPACK-marshalling library that supports [rocSOLVER](https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/) and cuSOLVER backends |
| {doc}`hipSPARSE <hipsparse:index>` | A SPARSE-marshalling library that supports [rocSPARSE](https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/) and cuSPARSE backends |
| {doc}`hipSPARSELt <hipsparselt:index>` | A SPARSE-marshalling library with multiple supported backends |
| {doc}`hipTensor <hiptensor:index>` | AMD's C++ library for accelerating tensor primitives based on the composable kernel library |
| [LLVM](https://github.com/ROCm/llvm-project) | A toolkit for the construction of highly optimized compilers, optimizers, and run-time environments |
| {doc}`MIGraphX <amdmigraphx:index>` | A graph inference engine that accelerates machine learning model inference |
| {doc}`MIOpen <miopen:index>` | An open source deep-learning library |
| [MIOpenGEMM](https://github.com/ROCm/MIOpenGEMM) | An OpenCL general matrix multiplication (GEMM) API and kernel generator |
| {doc}`MIVisionX <mivisionx:doxygen/html/index>` | A set of comprehensive computer vision and machine learning libraries, utilities, and applications |
| [Radeon Compute Profiler (RCP)](https://github.com/GPUOpen-Tools/radeon_compute_profiler/) | A performance analysis tool that gathers data from the API run-time and GPU for OpenCL and ROCm/HSA applications |
| {doc}`RCCL <rccl:index>` | A standalone library that provides multi-GPU and multi-node collective communication primitives |
| {doc}`rocAL <rocal:index>` | An augmentation library designed to decode and process images and videos |
| {doc}`rocALUTION <rocalution:index>` | A sparse linear algebra library for exploring fine-grained parallelism on ROCm runtime and toolchains |
| [RocBandwidthTest](https://github.com/ROCm/rocm_bandwidth_test/) | Captures the performance characteristics of buffer copying and kernel read/write operations |
| {doc}`rocBLAS <rocblas:index>` | A BLAS implementation (in the HIP programming language) on the ROCm runtime and toolchains |
| {doc}`rocDecode <rocdecode:index>` | A high performance video decode SDK for AMD GPUs |
| {doc}`rocFFT <rocfft:index>` | A software library for computing fast Fourier transforms (FFTs) written in HIP |
| [ROCK-Kernel-Driver](https://github.com/ROCm/ROCK-Kernel-Driver/) | An AMDGPU Driver with KFD that is used by ROCm |
| [ROCmCC](./reference/rocmcc.md) | A Clang/LLVM-based compiler |
| [ROCm cmake](https://github.com/ROCm/rocm-cmake) | A collection of CMake modules for common build and development tasks |
| {doc}`ROCm Data Center Tool <rdc:index>` | Simplifies administration and addresses key infrastructure challenges in AMD GPUs in cluster and data-center environments |
| [ROCm Debug Agent Library (ROCdebug-agent)](https://github.com/ROCm/rocr_debug_agent/) | A library that can print the state of all AMD GPU wavefronts that caused a queue error by sending a SIGQUIT signal to the process while the program is running |
| {doc}`ROCm debugger (ROCgdb) <rocgdb:index>` | A source-level debugger for Linux, based on the GNU Debugger (GDB) |
| {doc}`ROCdbgapi <rocdbgapi:index>` | The ROCm debugger API library |
| [rocminfo](https://github.com/ROCm/rocminfo/) | Reports system information |
| {doc}`ROCm Performance Primitives Library <rpp:index>` | A comprehensive high-performance computer vision library for AMD processors with HIP/OpenCL/CPU back-ends |
| {doc}`ROCm SMI <rocm_smi_lib:index>` | A C library for Linux that provides a user space interface for applications to monitor and control GPU applications |
| {doc}`ROCm Validation Suite <rocmvalidationsuite:index>` | A tool for detecting and troubleshooting common problems affecting AMD GPUs running in a high-performance computing environment |
| {doc}`rocPRIM <rocprim:index>` | A header-only library for HIP parallel primitives |
| {doc}`ROCProfiler <rocprofiler:profiler_home_page>` | A profiling tool for HIP applications |
| {doc}`rocRAND <rocrand:index>` | Provides functions that generate pseudorandom and quasirandom numbers |
| [ROCR-Runtime](https://github.com/ROCm/ROCR-Runtime/) | User-mode API interfaces and libraries necessary for host applications to launch compute kernels on available HSA ROCm kernel agents |
| {doc}`rocSOLVER <rocsolver:index>` | An implementation of LAPACK routines on ROCm software, implemented in the HIP programming language and optimized for AMD’s latest discrete GPUs |
| {doc}`rocSPARSE <rocsparse:index>` | Exposes a common interface that provides BLAS for sparse computation implemented on ROCm runtime and toolchains (in the HIP programming language) |
| {doc}`rocThrust <rocthrust:index>` | A parallel algorithm library |
| [ROCT-Thunk-Interface](https://github.com/ROCm/ROCT-Thunk-Interface/) | User-mode API interfaces used to interact with the ROCk driver |
| {doc}`ROCTracer <roctracer:index>` | Intercepts runtime API calls and traces asynchronous activity |
| {doc}`rocWMMA <rocwmma:index>` | A C++ library for accelerating mixed-precision matrix multiply-accumulate (MMA) operations |
| [Tensile](https://github.com/ROCm/Tensile) | A tool for creating benchmark-driven backend libraries for GEMMs, GEMM-like problems, and general N-dimensional tensor contractions |
| {doc}`TransferBench <transferbench:index>` | A utility to benchmark simultaneous transfers between user-specified devices (CPUs/GPUs) |

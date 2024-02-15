.. meta::
  :description: What is ROCm
  :keywords: ROCm projects, introduction, ROCm, AMD, runtimes, compilers, tools, libraries, API

***********************************************************
What is ROCm?
***********************************************************

ROCm is an open-source stack, composed primarily of open-source software, designed for
graphics processing unit (GPU) computation. ROCm consists of a collection of drivers, development
tools, and APIs that enable GPU programming from low-level kernel to end-user applications.

With ROCm, you can customize your GPU software to meet your specific needs. You can develop,
collaborate, test, and deploy your applications in a free, open source, integrated, and secure software
ecosystem. ROCm is particularly well-suited to GPU-accelerated high-performance computing (HPC),
artificial intelligence (AI), scientific computing, and computer aided design (CAD).

ROCm is powered by AMD’s
`Heterogeneous-computing Interface for Portability (HIP) <https://rocm.docs.amd.com/projects/HIP/en/latest/index.html>`_,
an open-source software C++ GPU programming environment and its corresponding runtime. HIP
allows ROCm developers to create portable applications on different platforms by deploying code on a
range of platforms, from dedicated gaming GPUs to exascale HPC clusters.

ROCm supports programming models, such as OpenMP and OpenCL, and includes all necessary open
source software compilers, debuggers, and libraries. ROCm is fully integrated into machine learning
(ML) frameworks, such as PyTorch and TensorFlow.

.. tip::
  If you're using Radeon GPUs, refer to the
  :doc:`Radeon-specific ROCm documentation <radeon:index>`.

ROCm project list
===============================================

ROCm consists of the following projects.

.. csv-table::
  :header: "Project", "Type", "Description", "License"

  "`AMD Compute Language Runtimes (CLR) <https://github.com/ROCm/clr>`_", "Runtime", "Contains source code for AMD's compute languages runtimes: :doc:`HIP <hip:index>` and OpenCL", "MIT"
  ":doc:`AMD SMI <amdsmi:index>`", "Tool", "C library for Linux that provides a user space interface for applications to monitor and control AMD devices", "MIT"
  "`AOMP <https://github.com/ROCm/aomp/>`_", "Compiler", "Scripted build of `LLVM <https://github.com/ROCm/llvm-project>`_ and supporting software", "Apache-2.0"
  "`Asynchronous Task and Memory Interface (ATMI) <https://github.com/ROCm/atmi/>`_", "Runtime", "Runtime framework for efficient task management in heterogeneous CPU-GPU systems", "MIT"
  ":doc:`Composable Kernel <composable_kernel:index>`", "Library (AI/ML)", "Provides a programming model for writing performance critical kernels for machine learning workloads across multiple architectures", "MIT"
  "`FLANG <https://github.com/ROCm/flang/>`_", "Compiler", "An out-of-tree Fortran compiler targeting LLVM", "Apache-2.0"
  "`half <https://github.com/ROCm/half/>`_", "Library (math)", "C++ header-only library that provides an IEEE 754 conformant, 16-bit half-precision floating-point type, along with corresponding arithmetic operators, type conversions, and common mathematical functions", "MIT"
  ":doc:`HIP <hip:index>`", "Runtime", AMD's GPU programming language extension and the GPU runtime", "MIT"
  ":doc:`hipBLAS <hipblas:index>`", "Library (math)", "BLAS-marshaling library that supports `rocBLAS <https://rocm.docs.amd.com/projects/rocBLAS/en/latest/>`_ and cuBLAS backends", "MIT"
  ":doc:`hipBLASLt <hipblaslt:index>`", "Library (math)", "Provides general matrix-matrix operations with a flexible API and extends functionalities beyond traditional BLAS library", "MIT"
  "`hipCC <https://github.com/ROCm/HIPCC>`_ ", "Compiler", "Compiler driver utility that calls Clang or NVCC and passes the appropriate include and library options for the target compiler and HIP infrastructure", "MIT"
  ":doc:`hipCUB <hipcub:index>`", "Library (C++ primitive)", "Thin header-only wrapper library on top of `rocPRIM <https://rocm.docs.amd.com/projects/rocPRIM/en/latest/>`_ or CUB that allows project porting using the CUB library to the HIP layer", "???"
  ":doc:`hipFFT <hipfft:index>`", "Library (math)", "Fast Fourier transforms (FFT)-marshalling library that supports rocFFT or cuFFT backends", "MIT"
  ":doc:`hipfort <hipfort:index>`", "Library (math)", "Fortran interface library for accessing GPU Kernels", "MIT"
  ":doc:`HIPIFY <hipify:index>`", "Compiler", "Translates CUDA source code into portable HIP C++", "MIT"
  ":doc:`hipRAND <hiprand:index>`", "Library (math)", "Ports CUDA applications that use the cuRAND library into the HIP layer", "MIT"
  ":doc:`hipSOLVER <hipsolver:index>`", "Library (math)", "An LAPACK-marshalling library that supports `rocSOLVER <https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/>`_ and cuSOLVER backends", "MIT"
  ":doc:`hipSPARSE <hipsparse:index>`", "Library (math)", "SPARSE-marshalling library that supports `rocSPARSE <https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/>`_ and cuSPARSE backends", "MIT"
  ":doc:`hipSPARSELt <hipsparselt:index>`", "Library (math)", "SPARSE-marshalling library with multiple supported backends", "???"
  ":doc:`hipTensor <hiptensor:index>`", "Library (C++ primitive)", "AMD's C++ library for accelerating tensor primitives based on the composable kernel library", "MIT"
  "`LLVM (amdclang) <https://github.com/ROCm/llvm-project>`_ ", "Compiler", "Toolkit for the construction of highly optimized compilers, optimizers, and run-time environments", "Apache-2.0"
  ":doc:`MIGraphX <amdmigraphx:index>`", "Library (AI/ML)", "Graph inference engine that accelerates machine learning model inference", "MIT"
  ":doc:`MIOpen <miopen:index>`", "Library (AI/ML)", "An open source deep-learning library", "MIT"
  ":doc:`MIVisionX <mivisionx:doxygen/html/index>`", "Library (AI/ML)", "Set of comprehensive computer vision and machine learning libraries, utilities, and applications", "MIT"
  "`Radeon Compute Profiler (RCP) <https://github.com/GPUOpen-Tools/radeon_compute_profiler/>`_ ", "Tool", "Performance analysis tool that gathers data from the API run-time and GPU for OpenCL and ROCm/HSA applications", "MIT"
  ":doc:`RCCL <rccl:index>`", "Library (communication)", "Standalone library that provides multi-GPU and multi-node collective communication primitives", "???"
  ":doc:`rocAL <rocal:index>`", "Library (AI/ML)", "An augmentation library designed to decode and process images and videos", "MIT"
  ":doc:`rocALUTION <rocalution:index>`", "Library (math)", "Sparse linear algebra library for exploring fine-grained parallelism on ROCm runtime and toolchains", "MIT"
  "`RocBandwidthTest <https://github.com/ROCm/rocm_bandwidth_test/>`_ ", "Tool", "Captures the performance characteristics of buffer copying and kernel read/write operations", "University of Illinois/NCSA"
  ":doc:`rocBLAS <rocblas:index>`", "Library (math)", "BLAS implementation (in the HIP programming language) on the ROCm runtime and toolchains", "???"
  ":doc:`rocFFT <rocfft:index>`", "Library (math)", "Software library for computing fast Fourier transforms (FFTs) written in HIP", "???"
  "`ROCmCC <./reference/rocmcc.md>`_ ", "Tool", "Clang/LLVM-based compiler", "???"
  "`ROCm CMake <https://github.com/ROCm/rocm-cmake>`_ ", "Tool", "Collection of CMake modules for common build and development tasks", "MIT"
  ":doc:`ROCm Data Center Tool <rdc:index>`", "Tool", "Simplifies administration and addresses key infrastructure challenges in AMD GPUs in cluster and data-center environments", "MIT"
  "`ROCm Debug Agent (ROCdebug-agent) <https://github.com/ROCm/rocr_debug_agent/>`_ ", "Tool", "Prints the state of all AMD GPU wavefronts that caused a queue error by sending a SIGQUIT signal to the process while the program is running", "University of Illinois/NCSA"
  ":doc:`ROCm debugger (ROCgdb) <rocgdb:index>`", "Tool", "Source-level debugger for Linux, based on the GNU Debugger (GDB)", "GPL-2.0"
  ":doc:`ROCdbgapi <rocdbgapi:index>`", "Tool", "ROCm debugger API library", "MIT"
  "`rocminfo <https://github.com/ROCm/rocminfo/>`_ ", "Tool", "Reports system information", "University of Illinois/NCSA"
  ":doc:`ROCm Performance Primitives (RPP) <rpp:index>`", "Library (AI/ML)", "Comprehensive high-performance computer vision library for AMD processors with HIP/OpenCL/CPU back-ends", "MIT"
  ":doc:`ROCm SMI <rocm_smi_lib:index>`", "Tool", "C library for Linux that provides a user space interface for applications to monitor and control GPU applications", "University of Illinois/NCSA"
  ":doc:`ROCm Validation Suite <rocmvalidationsuite:index>`", "Tool", "Detects and troubleshoots common problems affecting AMD GPUs running in a high-performance computing environment", "MIT"
  ":doc:`rocPRIM <rocprim:index>`", "Library (C++ primitive)", "Header-only library for HIP parallel primitives", "MIT"
  ":doc:`ROCProfiler <rocprofiler:profiler_home_page>`", "Tool", "Profiling tool for HIP applications", "MIT"
  ":doc:`rocRAND <rocrand:index>`", "Library (math)", "Provides functions that generate pseudorandom and quasirandom numbers", "MIT"
  "`ROCR-Runtime <https://github.com/ROCm/ROCR-Runtime/>`_ ", "Runtime", "User-mode API interfaces and libraries necessary for host applications to launch compute kernels on available HSA ROCm kernel agents", "University of Illinois/NCSA"
  ":doc:`rocSOLVER <rocsolver:index>`", "Library (math)", "An implementation of LAPACK routines on ROCm software, implemented in the HIP programming language and optimized for AMD's latest discrete GPUs", "???"
  ":doc:`rocSPARSE <rocsparse:index>`", "Library (math)", "Exposes a common interface that provides BLAS for sparse computation implemented on ROCm runtime and toolchains (in the HIP programming language)", "MIT"
  ":doc:`rocThrust <rocthrust:index>`", "Library (C++ primitive)", "Parallel algorithm library", "Apache-2.0"
  ":doc:`ROCTracer <roctracer:index>`", "Tool", "Intercepts runtime API calls and traces asynchronous activity", "MIT"
  ":doc:`rocWMMA <rocwmma:index>`", "Library (math)", "C++ library for accelerating mixed-precision matrix multiply-accumulate (MMA) operations", "MIT"
  "`Tensile <https://github.com/ROCm/Tensile>`_ ", "Library (math)", "Creates benchmark-driven backend libraries for GEMMs, GEMM-like problems, and general N-dimensional tensor contractions", "MIT"
  ":doc:`TransferBench <transferbench:index>`", "Tool", "Utility to benchmark simultaneous transfers between user-specified devices (CPUs/GPUs)", "MIT"

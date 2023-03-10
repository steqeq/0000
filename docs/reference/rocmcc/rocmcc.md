# Introduction to Compiler Reference Guide

ROCmCC is a Clang/LLVM-based compiler. It is optimized for high-performance computing on AMD GPUs and CPUs and supports various heterogenous programming models such as HIP, OpenMP, and OpenCL.

ROCmCC is made available via two packages: rocm-llvm and rocm-llvm-alt. The differences are shown in this table:

| <b>Table 1. rocm-llvm vs. rocm-llvm-alt</b>|
| rocm-llvm | rocm-llvm-alt |
| ----------- | ----------- |
| Installed by default when ROCm™ itself is installed | An optional package |
| Provides an open-source compiler | Provides an additional closed-source compiler for users interested in additional CPU optimizations not available in rocm-llvm |

For more details, follow this table:

| <b>Table 2. Details Table</b>|
| For | See |
| ----------- | ----------- |
| The latest usage information for AMD GPU |[https://llvm.org/docs/AMDGPUUsage.html](https://llvm.org/docs/AMDGPUUsage.html) |
|Usage information for a specific ROCm release | [https://llvm.org/docs/AMDGPUUsage.html] (https://llvm.org/docs/AMDGPUUsage.html)|
| Source code for rocm-llvm | [https://github.com/RadeonOpenCompute/llvm-project](https://github.com/RadeonOpenCompute/llvm-project) |

## ROCm Compiler Interfaces
ROCm currently provides two compiler interfaces for compiling HIP programs:
- /opt/rocm/bin/hipcc
- /opt/rocm/bin/amdclang++

Both leverage the same LLVM compiler technology with the AMD GCN GPU support; however, they offer a slightly different user experience. The hipcc command-line interface aims to provide a more familiar user interface to users who are experienced in CUDA but relatively new to the ROCm/HIP development environment. On the other hand, amdclang++ provides a user interface identical to the clang++ compiler. It is more suitable for experienced developers who want to directly interact with the clang compiler and gain full control of their application’s build process.

The major differences between hipcc and amdclang++ are listed below:

| <b>Table 3. Differences Between hipcc and amdclang++</b>|
|| Hipcc | amdclang++ |
| ----------- | ----------- | ----------- |
| Compiling HIP source files | Treats all source files as HIP language source files | Enables the HIP language support for files with the “.hip” extension or through the -x hip compiler option |
| Automatic GPU architecture detection | Auto-detects the GPUs available on the system and generates code for those devices when no GPU architecture is specified | Has AMD GCN gfx803 as the default GPU architecture. The --offload-arch compiler option may be used to target other GPU architectures |
| Finding a HIP installation | Finds the HIP installation based on its own location and its knowledge about the ROCm directory structure | First looks for HIP under the same parent directory as its own LLVM directory and then falls back on /opt/rocm. Users can use the --rocm-path option to instruct the compiler to use HIP from the specified ROCm installation. |
| Linking to the HIP runtime library | Is configured to automatically link to the HIP runtime from the detected HIP installation | Requires the --hip-link flag to be specified to link to the HIP runtime. Alternatively, users can use the -l\<dir\> -lamdhip64 option to link to a HIP runtime library. |
| Device function inlining | Inlines all GPU device functions, which provide greater performance and compatibility for codes that contain file scoped or device function scoped \_\_shared\_\_ variables. However, it may increase compile time. | Relies on inlining heuristics to control inlining. Users experiencing performance or compilation issues with code using file scoped or device function scoped \_\_shared\_\_ variables could try -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false to work around the issue. There are plans to address these issues with future compiler improvements. |
| Source code location | Developed at [https://github.com/ROCm-Developer-Tools/HIPCC](https://github.com/ROCm-Developer-Tools/HIPCC) | Developed at [https://github.com/RadeonOpenCompute/llvm-project](https://github.com/RadeonOpenCompute/llvm-project) |



# Compiler Options and Features

This chapter discusses compiler options and features.

## AMD GPU Compilation
This table provides the most commonly used compiler options for GPU code.

| <b>Table 4. Compiler Options</b>|
| Option | Description |
| ----------- | ----------- |
| -x hip | Compiles the source file as a HIP program |

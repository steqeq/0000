# Introduction to Compiler Reference Guide

ROCmCC is a Clang/LLVM-based compiler. It is optimized for high-performance computing on AMD GPUs and CPUs and supports various heterogenous programming models such as HIP, OpenMP, and OpenCL.

ROCmCC is made available via two packages: rocm-llvm and rocm-llvm-alt. The differences are shown in this table:
||
|:--:|
| <b>Table 1. rocm-llvm vs. rocm-llvm-alt</b>|
||

| rocm-llvm | rocm-llvm-alt |
| ----------- | ----------- |
| Installed by default when ROCm™ itself is installed | An optional package |
| Provides an open-source compiler | Provides an additional closed-source compiler for users interested in additional CPU optimizations not available in rocm-llvm |

For more details, follow this table:

||
|:--:|
| <b>Table 2. Details Table</b>|
||

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
||
|:--:|
| <b>Table 3. Differences Between hipcc and amdclang++</b>|
||

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

||
|:--:|
| <b>Table 4. Compiler Options</b>|
||

| Option | Description |
| ----------- | ----------- |
| -x hip | Compiles the source file as a HIP program |
| -fopenmp | Enables the OpenMP support |
| -fopenmp-targets=\<gpu\> | Enables the OpenMP target offload support of the specified GPU architecture |
| --gpu-max-threads-per-block=\<value\> | Sets default launch bounds for kernels |
| -munsafe-fp-atomics | Enables unsafe floating point atomic instructions (AMDGPU only) |
| -ffast-math | Allows aggressive, lossy floating-point optimizations |
| -mwavefrontsize64/-mno-wavefrontsize64 | Sets wavefront size to be 64 or 32 on RDNA architectures |
| -mcumode | Switches between CU and WGP modes on RDNA architectures |
| --offload-arch=\<gpu\> | HIP offloading target ID in the form of a device architecture followed by target ID features delimited by a colon. Each target ID feature is a predefined string followed by a plus or minus sign (e.g. gfx908:xnack+:sramecc-). May be specified more than once |
| -g | Generates source-level debug information |
| -fgpu-rdc/-fno-gpu-rdc | Generates relocatable device code, also known as separate compilation mode |

## AMD Optimizations for Zen Architectures
The CPU compiler optimizations described in this chapter originate from the AMD Optimizing C/C++ Compiler (AOCC) compiler. They are available in ROCmCC if the optional rocm-llvm-alt package is installed. The user’s interaction with the compiler does not change once rocm-llvm-alt is installed. The user should use the same compiler entry point, provided AMD provides high-performance compiler optimizations for Zen-based processors in AOCC. 

For more information, refer to [https://developer.amd.com/amd-aocc/](https://developer.amd.com/amd-aocc/).

### -famd-opt
Enables a default set of AMD proprietary optimizations for the AMD Zen CPU architectures.

-fno-amd-opt disables the AMD proprietary optimizations.

The -famd-opt flag is useful when a user wants to build with the proprietary optimization compiler and not have to depend on setting any of the other proprietary optimization flags.

:::{note}
-famd-opt can be used in addition to the other proprietary CPU optimization flags. The table of optimizations below implicitly enables the invocation of the AMD proprietary optimizations compiler, whereas the -famd-opt flag requires this to be handled explicitly.
:::

### -fstruct-layout=[1,2,3,4,5,6,7]
Analyzes the whole program to determine if the structures in the code can be peeled and the pointer or integer fields in the structure can be compressed. If feasible, this optimization transforms the code to enable these improvements. This transformation is likely to improve cache utilization and memory bandwidth. It is expected to improve the scalability of programs executed on multiple cores.

This is effective only under flto, as the whole program analysis is required to perform this optimization. Users can choose different levels of aggressiveness with which this optimization can be applied to the application, with 1 being the least aggressive and 7 being the most aggressive level.

||
|:--:|
| <b>Table 5. -fstruct-layout Values and Their Effects</b>|
||

| -fstruct-layout value | Structure peeling | Pointer size after selective compression of self-referential pointers in structures, wherever safe | Type of structure fields eligible for compression | Whether compression performed under safety check |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| 1 | Enabled | NA | NA | NA |
| 2 | Enabled | 32-bit | NA | NA |
| 3 | Enabled | 16-bit | NA | NA |
| 4 | Enabled | 32-bit | Integer | Yes |
| 5 | Enabled | 16-bit | Integer | Yes |
| 6 | Enabled | 32-bit | 64-bit signed int or unsigned int. Users must ensure that the values assigned to 64-bit signed int fields are in range -(2^31 - 1) to +(2^31 - 1) and 64-bit unsigned int fields are in the range 0 to +(2^31 - 1). Otherwise, you may obtain incorrect results. | No. Users must ensure the safety based on the program compiled. |
| 7 | Enabled | 16-bit | 64-bit signed int or unsigned int. Users must ensure that the values assigned to 64-bit signed int fields are in range -(2^31 - 1) to +(2^31 - 1) and 64-bit unsigned int fields are in the range 0 to +(2^31 - 1). Otherwise, you may obtain incorrect results. | No. Users must ensure the safety based on the program compiled. |

### -fitodcalls
Promotes indirect-to-direct calls by placing conditional calls. Application or benchmarks that have a small and deterministic set of target functions for function pointers passed as call parameters benefit from this optimization. Indirect-to-direct call promotion transforms the code to use all possible determined targets under runtime checks and falls back to the original code for all the other cases. Runtime checks are introduced by the compiler for each of these possible function pointer targets followed by direct calls to the targets.

This is a link time optimization, which is invoked as -flto -fitodcalls

### -fitodcallsbyclone
Performs value specialization for functions with function pointers passed as an argument. It does this specialization by generating a clone of the function. The cloning of the function happens in the call chain as needed, to allow conversion of indirect function call to direct call.

This complements -fitodcalls optimization and is also a link time optimization, which is invoked as -flto -fitodcallsbyclone.

### -fremap-arrays
Transforms the data layout of a single dimensional array to provide better cache locality. This optimization is effective only under flto, as the whole program needs to be analyzed to perform this optimization, which can be invoked as -flto -fremap-arrays.
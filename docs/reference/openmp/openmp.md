# OpenMP Support in ROCm

## Introduction to OpenMP Support Guide

he ROCm™ installation includes an LLVM-based implementation that fully supports the OpenMP 4.5 standard and a subset of OpenMP 5.0, 5.1, and 5.2 standards. Fortran, C/C++ compilers, and corresponding runtime libraries are included. Along with host APIs, the OpenMP compilers support offloading code and data onto GPU devices. This document briefly describes the installation location of the OpenMP toolchain, example usage of device offloading, and usage of rocprof with OpenMP applications. The GPUs supported are the same as those supported by this ROCm release. See the list of supported GPUs in the installation guide at [https://docs.amd.com/](https://docs.amd.com/).

### Installation

The OpenMP toolchain is automatically installed as part of the standard ROCm installation and is available under /opt/rocm-{version}/llvm. The sub-directories are:

bin: Compilers (flang and clang) and other binaries.

- examples: The usage section below shows how to compile and run these programs.

- include: Header files.

- lib: Libraries including those required for target offload.

- lib-debug: Debug versions of the above libraries.

## OpenMP: Usage

The example programs can be compiled and run by pointing the environment variable AOMP to the OpenMP install directory.

**Example:**

```bash
% export AOMP=/opt/rocm-{version}/llvm
% cd $AOMP/examples/openmp/veccopy
% make run
```

The above invocation of Make compiles and runs the program. Note the options that are required for target offload from an OpenMP program:

```bash
-target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=<gpu-arch>
```

Obtain the value of gpu-arch by running the following command:

```bash
% /opt/rocm-{version}/bin/rocminfo | grep gfx
```

[//]: # (dated link below, needs upading)

See the complete list of compiler command-line references [here](https://github.com/RadeonOpenCompute/llvm-project/blob/amd-stg-open/clang/docs/ClangCommandLineReference.rst).

### Using rocprof with OpenMP

The following steps describe a typical workflow for using rocprof with OpenMP code compiled with AOMP:

1. Run rocprof with the program command line:

    ```bash
    % rocprof <application> <args>
    ```

    This produces a results.csv file in the user’s current directory that shows basic stats such as kernel names, grid size, number of registers used, etc. The user can choose to specify the preferred output file name using the o option.

2. Add options for a detailed result:

    ```bash
    --stats: % rocprof --stats <application> <args>
    ```

    The stats option produces timestamps for the kernels. Look into the output CSV file for the field, DurationNs, which is useful in getting an understanding of the critical kernels in the code.

    Apart from --stats, the option --timestamp on produces a timestamp for the kernels.

3. After learning about the required kernels, the user can take a detailed look at each one of them. rocprof has support for hardware counters: a set of basic and a set of derived ones. See the complete list of counters using options --list-basic and --list-derived. rocprof accepts either a text or an XML file as an input.

For more details on rocprof, refer to the ROCm Profiling Tools document on [https://docs.amd.com](https://docs.amd.com).

### Using Tracing Options

**Prerequisite:** When using the --sys-trace option, compile the OpenMP program with:

```bash
    -Wl,–rpath,/opt/rocm-{version}/lib -lamdhip64 
```

The following tracing options are widely used to generate useful information:

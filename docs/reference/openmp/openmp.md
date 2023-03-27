# OpenMP Support in ROCm

Pull from
<https://docs.amd.com/bundle/OpenMP-Support-Guide-v5.4/page/Introduction_to_OpenMP_Support_Guide.html>

[//]: # (Unifying "Introduction to OpenMP Support Guide", "OpenMP: Usage" and "OpenMP: Features" into a single page, excluding "Legal Disclaimer and Copyright Information" to be handled centrally)

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

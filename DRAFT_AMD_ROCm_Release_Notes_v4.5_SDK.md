![A picture containing drawing Description automatically
generated](./images//media/AMD_logo.png){width="1.1716666666666666in"
height="0.475in"}

+-----------------------------------------------------------------------+
| DRAFT                                                                 |
|                                                                       |
| AMD ROCm™ v4.5                                                        |
|                                                                       |
| Release Notes                                                         |
+-----------------------------------------------------------------------+

  ---------- ---------- --------- ------ -- ----- --- ---- ---- ---- ---- ----- ----------
             Revision             1027                                          

             Issue      October                                                 
             Date:      2021                                                    

                                                                                
  ---------- ---------- --------- ------ -- ----- --- ---- ---- ---- ---- ----- ----------

© 2021 Advanced Micro Devices, Inc. All Rights Reserved.

**DISCLAIMER\
**

The information presented in this document is for informational purposes
only and may contain technical inaccuracies, omissions, and
typographical errors. The information contained herein is subject to
change and may be rendered inaccurate for many reasons, including but
not limited to product and roadmap changes, component and motherboard
versionchanges, new model and/or product releases, product differences
between differing manufacturers, software changes, BIOS flashes,
firmware upgrades, or the like. Any computer system has risks of
security vulnerabilities that cannot be completely prevented or
mitigated.AMD assumes no obligation to update or otherwise correct or
revise this information. However, AMD reserves the right to revise this
information and to make changes from time to time to the content hereof
without obligation of AMD to notify any person of such revisions or
changes.THIS INFORMATION IS PROVIDED 'AS IS." AMD MAKES NO
REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND
ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS
THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY
IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR
ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR
ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES
ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS
EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.AMD, the AMD Arrow
logo,\[insert all other AMD trademarks used in the material here perAMD
Trademarks\]and combinations thereof are trademarks of Advanced Micro
Devices, Inc.Other product names used in this publication are for
identification purposes only and may be trademarks of their respective
companies. \[Insert any third party trademark attribution here per
AMD\'sThird Party Trademark List.\]

©\[2021\]Advanced Micro Devices, Inc.All rights reserved.

Disclaimer

Third-party content is licensed to you directly by the third party that
owns the content and is not licensed to you by AMD. ALL LINKED
THIRD-PARTY CONTENT IS PROVIDED "AS IS" WITHOUT A WARRANTY OF ANY KIND.
USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND
UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY
CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES
THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.

\[This page is left blank intentionally\]

[]{#_Toc503941611 .anchor}Table of Contents

[Table of Contents [4](#_Toc503941611)](#_Toc503941611)

[ROCm Installation Updates
[6](#rocm-installation-updates)](#rocm-installation-updates)

[List of Supported Operating Systems
[6](#list-of-supported-operating-systems)](#list-of-supported-operating-systems)

[MIOpen Supports AMD Radeon™ RX 6800
[6](#miopen-supports-amd-radeon-rx-6800)](#miopen-supports-amd-radeon-rx-6800)

[Enhanced Installation Process for ROCm v4.5
[7](#enhanced-installation-process-for-rocm-v4.5)](#enhanced-installation-process-for-rocm-v4.5)

[AMD ROCm V4.5 Documentation Updates -- WORK in PROGRESS
[8](#amd-rocm-v4.5-documentation-updates-work-in-progress)](#amd-rocm-v4.5-documentation-updates-work-in-progress)

[AMD ROCm Installation Guide
[8](#amd-rocm-installation-guide)](#amd-rocm-installation-guide)

[HIP Documentation Updates
[8](#hip-documentation-updates)](#hip-documentation-updates)

[System Interface Management
[8](#system-interface-management)](#system-interface-management)

[ROCm SMI API Guide [8](#rocm-smi-api-guide)](#rocm-smi-api-guide)

[ROC Debugger User and API Guide
[9](#roc-debugger-user-and-api-guide)](#roc-debugger-user-and-api-guide)

[AMD ROCm General Documentation Links
[9](#amd-rocm-general-documentation-links)](#amd-rocm-general-documentation-links)

[What's New in This Release
[10](#whats-new-in-this-release)](#whats-new-in-this-release)

[HIP Enhancements [10](#hip-enhancements)](#hip-enhancements)

[HIP Direct Dispatch [10](#hip-direct-dispatch)](#hip-direct-dispatch)

[Support for HIP Graph
[10](#support-for-hip-graph)](#support-for-hip-graph)

[Enhanced launch_bounds Check Error Log Message
[10](#enhanced-launch_bounds-check-error-log-message)](#enhanced-launch_bounds-check-error-log-message)

[HIP Runtime Compilation
[11](#hip-runtime-compilation)](#hip-runtime-compilation)

[New Flag for Backwards Compatibility on float/double atomicAdd Function
[11](#new-flag-for-backwards-compatibility-on-floatdouble-atomicadd-function)](#new-flag-for-backwards-compatibility-on-floatdouble-atomicadd-function)

[Updated HIP Version Definition
[12](#updated-hip-version-definition)](#updated-hip-version-definition)

[Planned HIP Enhancements and Fixes
[13](#planned-hip-enhancements-and-fixes)](#planned-hip-enhancements-and-fixes)

[Unified Memory Support in ROCm
[13](#unified-memory-support-in-rocm)](#unified-memory-support-in-rocm)

[Supported Operating Systems and Versions
[13](#supported-operating-systems-and-versions)](#supported-operating-systems-and-versions)

[Unified Memory Support and XNACK
[14](#unified-memory-support-and-xnack)](#unified-memory-support-and-xnack)

[System Management Interface
[15](#system-management-interface)](#system-management-interface)

[Enhanced ROCm SMI --setpoweroverdrive Functionality
[15](#enhanced-rocm-smi-setpoweroverdrive-functionality)](#enhanced-rocm-smi-setpoweroverdrive-functionality)

[OpenMP Enhancements [16](#openmp-enhancements)](#openmp-enhancements)

[ROCm Math and Communication Libraries
[17](#rocm-math-and-communication-libraries)](#rocm-math-and-communication-libraries)

[Known Issues in This Release
[23](#known-issues-in-this-release)](#known-issues-in-this-release)

[Compiler Support for Function Pointers and Virtual Functions
[23](#compiler-support-for-function-pointers-and-virtual-functions)](#compiler-support-for-function-pointers-and-virtual-functions)

[Debugger Process Exit May Cause ROCgdb Internal Error
[23](#debugger-process-exit-may-cause-rocgdb-internal-error)](#debugger-process-exit-may-cause-rocgdb-internal-error)

[Deprecations [23](#deprecations)](#deprecations)

[AMD Instinct MI25 End of Life
[23](#amd-instinct-mi25-end-of-life)](#amd-instinct-mi25-end-of-life)

[Hardware and Software Support
[24](#hardware-and-software-support)](#hardware-and-software-support)

[Hardware Support [24](#hardware-support)](#hardware-support)

[Supported Graphics Processing Units
[24](#supported-graphics-processing-units)](#supported-graphics-processing-units)

[Supported CPUs [26](#supported-cpus)](#supported-cpus)

[Not supported or limited support under ROCm
[27](#not-supported-or-limited-support-under-rocm)](#not-supported-or-limited-support-under-rocm)

[ROCm Support in Upstream Linux Kernels
[29](#rocm-support-in-upstream-linux-kernels)](#rocm-support-in-upstream-linux-kernels)

# ROCm Installation Updates

This document describes the features, fixed issues, and information
about downloading and installing the AMD ROCm™ software.

It also covers known issues and deprecations in this release.

## List of Supported Operating Systems

The AMD ROCm platform supports the following operating systems:

  -----------------------------------------------------------------------
  OS                       Kernel 
  ------------------------ ----------------------------------------------
  SLES15 SP3               5.3.18-24.49

  RHEL 7.9                 3.10.0-1160.6.1.el7

  CentOS 7.9               3.10.0-1127

  RHEL 8.4                 4.18.0-193.1.1.el8

  CentOS 8.3               4.18.0-193.el8

  Ubuntu 18.04.5           5.4.0-71-generic

  Ubuntu 20.04.3HWE        5.8.0-48-generic

  Host OS                  Azure RS1.86

  Guest OS                 Ubuntu 20.04
  -----------------------------------------------------------------------

### MIOpen Supports AMD Radeon™ RX 6800

MIOpen now extends support to AMD Radeon™ RX 6800.

## Enhanced Installation Process for ROCm v4.5

In addition to the installation method using the native Package Manager,
AMD ROCm v4.5 introduces added methods to install ROCm. With this
release, the ROCm installation uses the *amdgpu-install* and
*amdgpu-uninstall* scripts. 

The *amdgpu-install* script streamlines the installation process by:

-   Abstracting the distribution-specific package installation logic

-   Performing the repository set-up

-   Allowing user to specify the use case and automating the
    installation of all the required packages,

-   Performing post-install checks to verify whether the installation
    was performed successfully

-   Installing the uninstallation script

The *amdgpu-uninstall* script allows the removal of the entire ROCm
stack by using a single command.

Some of the ROCm-specific use cases that the installer currently
supports are:

-   OpenCL (ROCr/KFD based) runtime

-   HIP runtimes

-   ROCm libraries and applications

-   ROCm Compiler and device libraries

-   ROCr runtime and thunk

For more information, refer to the [Installation
Methods](#_Installation_Methods) section in this guide.

**Note:** Graphics use cases are not supported in this release.

For more details, refer to the AMD ROCm Installation Guide v4.5 at,

**Add link**

# AMD ROCm V4.5 Documentation Updates -- WORK in PROGRESS

## AMD ROCm Installation Guide 

The AMD ROCm Installation Guide in this release includes the following
updates:

-   New - Installation Guide for ROCm v4.5

-   HIP installation instructions

## HIP Documentation Updates

-   HIP Programming Guide

-   HIP API Guide

-   HIP-Supported CUDA API Reference Guide

-   AMD ROCm Compiler Reference Guide

-   HIP FAQ

> <https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-FAQ.html#hip-faq>

## System Interface Management

-   System Interface Management (SMI)

## ROCm SMI API Guide

-   ROCm SMI API Guide

## ROC Debugger User and API Guide 

-   ROCDebugger User Guide

> <https://github.com/RadeonOpenCompute/ROCm/blob/master/AMD_ROCDebugger_User_Guide.pdf>

-   Debugger API Guide

> https://github.com/RadeonOpenCompute/ROCm/blob/master/AMD_ROCDebugger_API.pdf

## AMD ROCm General Documentation Links

-   For AMD ROCm documentation, see

> [[https://rocmdocs.amd.com/en/latest/]{.underline}](https://rocmdocs.amd.com/en/latest/)

-   For installation instructions on supported platforms, see

> [[https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html]{.underline}](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html)

-   For AMD ROCm binary structure, see

> [[https://rocmdocs.amd.com/en/latest/Installation_Guide/Software-Stack-for-AMD-GPU.html]{.underline}](https://rocmdocs.amd.com/en/latest/Installation_Guide/Software-Stack-for-AMD-GPU.html)

-   For AMD ROCm release history, see

> https://rocmdocs.amd.com/en/latest/Current_Release_Notes/ROCm-Version-History.html

# What's New in This Release 

## HIP Enhancements

The ROCm v4.5 release consists of the following HIP enhancements:

### HIP Direct Dispatch

The conventional producer-consumer model where the host thread(producer)
enqueues commands to a command queue (per stream), which is then
processed by a separate, per-stream worker thread (consumer) created by
the runtime, is no longer applicable.

In this release, for Direct Dispatch, the runtime directly queues a
packet to the AQL queue (user mode queue to GPU) in Dispatch and some of
the synchronization. This new functionality indicates the total latency
of the HIP Dispatch API and the latency to launch the first wave on the
GPU.

In addition, eliminating the threads in runtime has reduced the variance
in the dispatch numbers as the thread scheduling delays and
atomics/locks synchronization latencies are reduced.

This feature can be disabled by setting the following environment
variable,

AMD_DIRECT_DISPATCH=0

### Support for HIP Graph 

ROCm v4.5 extends support for HIP Graph. For details, refer to the HIP
API Guide at

**Add link**

### Enhanced *launch_bounds* Check Error Log Message 

When a kernel is launched with HIP APIs, for example,
hipModuleLaunchKernel(), HIP validates to check that input kernel
dimension size is not larger than specified launch_bounds.

If exceeded, HIP returns launch failure if AMD_LOG_LEVEL is set with the
proper value. Users can find more information in the error log message,
including launch parameters of kernel dim size, launch bounds, and the
name of the faulting kernel. It is helpful to figure out the faulting
kernel. Besides, the kernel dim size and launch bounds values will also
assist in debugging such failures.

For more details, refer to the HIP Programming Guide at

**Add link**

### HIP Runtime Compilation

HIP now supports runtime compilation (hipRTC), the usage of which will
provide the possibility of optimizations and performance improvement
compared with other APIs via regular offline static compilation.

hipRTC APIs accept HIP source files in character string format as input
parameters and create handles of programs by compiling the HIP source
files without spawning separate processes.

For more details on hipRTC APIs, refer to the HIP API Guide at

**Add link**

### New Flag for Backwards Compatibility on float/double atomicAdd Function

In the ROCm4.5 release, a new compilation flag is introduced as an
option in the CMAKE file. This flag ensures backwards compatibility in
float/double atomicAdd functions.

\_\_HIP_USE_CMPXCHG_FOR_FP_ATOMICS

This compilation flag is not set("0") by default, so the HIP runtime
uses the current float/double atomicAdd functions.

If this compilation flag is set to "1" with the CMAKE option, the
existing float/double atomicAdd functions is used for compatibility with
compilers that do not support floating point atomics.

D\_\_HIP_USE_CMPXCHG_FOR_FP_ATOMICS=1

For details on how to build the HIP runtime, refer to the HIP
Programming Guide at

**Add link**

**\
**

### Updated HIP Version Definition

The HIP version definition is updated as follows:

HIP_VERSION=HIP_VERSION_MAJOR \* 10000000 + HIP_VERSION_MINOR \*
100000 + HIP_VERSION_PATCH)

The HIP version can be queried from the following HIP API call,

hipRuntimeGetVersion(&runtimeVersion);

The version returned is always greater than the versions in the previous
ROCm releases.

**Note:** The version definition of the HIP runtime is different from
that of CUDA. The function returns the HIP runtime version on the AMD
platform, while on the NVIDIA platform, it returns the CUDA runtime
version. There is no mapping or a correlation between the HIP and CUDA
versions.

**\
**

### Planned HIP Enhancements and Fixes

#### changes to hiprtc implementation to match nvrtc behavior

In this release, there are changes to the *hiprtc* implementation to
match the *nvrtc* behavior.

**Impact:** Applications can no longer explicitly include HIP runtime
header files. Minor code changes are required to remove the HIP runtime
header files.

#### HIP device attribute enumeration

In a future release, there will be a breaking change in the HIP device
attribute enumeration. Enum values are being rearranged to accommodate
future enhancements and additions.

**Impact:** This will require users to rebuild their applications. No
code changes are required.

#### changes to behavior of hipGetLastError() and hipPeekAtLastError() to match CUDA behavior available

In a later release, changes to behavior of hipGetLastError() and
hipPeekAtLastError() to match CUDA behavior will be available.

**Impact:** Applications relying on the previous behavior will be
impacted and may require some code changes.

## Unified Memory Support in ROCm

Unified memory allows applications to map and migrate data between CPU
and GPU seamlessly without explicitly copying it between different
allocations. This enables a more complete implementation of
*hipMallocManaged*, *hipMemAdvise*, *hipMemPrefetchAsync* and related
APIs. Without unified memory, these APIs only support system memory.
With unified memory, the driver can automatically migrate such memory to
GPU memory for faster access.

### Supported Operating Systems and Versions

This feature is only supported on recent Linux kernels. Currently, it
works on Ubuntu versions with 5.6 or newer kernels and the DKMS driver
from ROCm. Current releases of RHEL and SLES do not support this feature
yet. Future releases of those distributions will add support for this.
The unified memory feature is also supported in the KFD driver included
with upstream kernels starting from Linux 5.14.

Unified memory only works on GFXv9 and later GPUs, including Vega10 and
MI100. Fiji, Polaris and older GPUs are not supported. To check whether
unified memory is enabled, look in the kernel log for this message:

\$ dmesg \| grep \"HMM registered"

If unified memory is enabled, there should be a message like "HMM
registered xyzMB device memory". If unified memory is not supported on
your GPU or kernel version, this message is missing.

### Unified Memory Support and XNACK

Unified memory support comes in two flavours, XNACK-enabled and
XNACK-disabled. XNACK refers to the ability of the GPU to handle page
faults gracefully and retry a memory access. In XNACK-enabled mode, the
GPU can handle retry after page-faults, which enables mapping and
migrating data on demand, as well as memory overcommitment. In
XNACK-disabled mode, all memory must be resident and mapped in the GPU
page tables when the GPU is executing application code. Any migrations
involve temporary preemption of the GPU queues by the driver. Both page
fault handling and preemptions, happen automatically and are transparent
to the applications.

XNACK-enabled mode only has experimental support. XNACK-enabled mode
requires compiling shader code differently. By default, the ROCm
compiler builds code that works in both modes. Code can be optimized for
one specific mode with compiler options:

OpenCL:

clang \... -mcpu=gfx908:***xnack+***:sramecc- \... // xnack on, sramecc
off\
clang \... -mcpu=gfx908:***xnack-***:sramecc+ \... // xnack off, sramecc
on

HIP:

clang \... \--cuda-gpu-arch=gfx906:xnack+ \... // xnack on\
clang \... \--cuda-gpu-arch=gfx906:xnack- \... // xnack off

Not all the math libraries included in ROCm support XNACK-enabled mode
on current hardware. Applications will fail to run if their shaders are
compiled in the incorrect mode.

On current hardware, the XNACK mode can be chosen at boot-time by a
module parameter amdgpu.noretry. The default is XNACK-disabled
(amdgpu.noretry=1).

## System Management Interface 

### Enhanced ROCm SMI --setpoweroverdrive Functionality

The ROCm System Management Interface (SMI) *--setpoweroverdrive*
functionality is used to lower the power cap on a device without needing
to enable the OverDrive functionality in the driver. Similarly, even
with the OverDrive driver functionality enabled, it is possible to
request a lower power cap than the card's default.

Currently, any use of the *--setpoweroverdrive* functionality in
rocm-smi prints an out-of-spec warning to the screen and requires the
user to agree that using this functionality potentially voids their
warranty. However, this warning should only be printed when users are
trying to set the power cap to higher than the card's default, which
requires the OverDrive driver functionality to be enabled.

For example:

The default power cap is 225.0W before any changes.

\[atitest\@rhel85 smi\]\$ ./rocm_smi.py --resetpoweroverdrive

======================= ROCm System Management Interface
=======================

========================== Reset GPU Power OverDrive
===========================

GPU\[0\] : Successfully reset Power OverDrive to: 225W

============================ End of ROCm SMI Log
============================

Now, after using --setpoweroverdrive to lower the power cap to 123
watts:

\[atitest\@rhel85 smi\]\$ ./rocm_smi.py --setpoweroverdrive 123

======================= ROCm System Management Interface
=======================

=========================== Set GPU Power OverDrive
============================

GPU\[0\] : Successfully set power to: 123W

======================= End of ROCm SMI Log
==============================

Setting a power cap lower than the default of 225.0W (in this case,
123W) does not give a warning.

To verify that the power is set to the correct value:

\[atitest\@rhel85 smi\]\$ ./rocm_smi.py --showmaxpower

======================= ROCm System Management Interface
=======================

======================== Power Cap ===================================

GPU\[0\] : Max Graphics Package Power (W): 123.0

========================End of ROCm SMI Log
==============================

## OpenMP Enhancements

The ROCm installation includes an LLVM-based implementation, which fully
supports OpenMP 4.5 standard and a subset of the OpenMP 5.0 standard.
Fortran and C/C++ compilers and corresponding runtime libraries are
included. Along with host APIs, the OpenMP compilers support offloading
code and data onto GPU devices.

For more information, refer to

<https://rocmdocs.amd.com/en/latest/Programming_Guides/openmp_support.html>

## ROCm Math and Communication Libraries

In this release, ROCm Math and Communication Libraries consists of the
following enhancements and fixes:

+-----------+----------------------------------------------------------+
| Library   | Changes                                                  |
+===========+==========================================================+
| rocBLAS   | **Optimizations**                                        |
|           |                                                          |
|           | -   Improved performance of non-batched and batched syr  |
|           |     for all sizes and data types                         |
|           |                                                          |
|           | -   Improved performance of non-batched and batched hemv |
|           |     for all sizes and data types                         |
|           |                                                          |
|           | -   Improved performance of non-batched and batched symv |
|           |     for all sizes and data types                         |
|           |                                                          |
|           | -   Improved memory utilization in rocblas-bench,        |
|           |     rocblas-test gemm functions, increasing possible     |
|           |     runtime sizes.                                       |
|           |                                                          |
|           | **Changes**                                              |
|           |                                                          |
|           | -   Update from C++14 to C++17.                          |
|           |                                                          |
|           | -   Packaging split into a runtime package (called       |
|           |     rocblas) and a development package (called           |
|           |     rocblas-dev for .deb packages, and rocblas-devel for |
|           |     .rpm packages). The development package depends on   |
|           |     runtime. The runtime package suggests the            |
|           |     development package for all supported OSes except    |
|           |     CentOS 7 to aid in the transition. The suggested     |
|           |     feature in packaging is introduced as a deprecated   |
|           |     feature and will be removed in a future ROCm         |
|           |     release.                                             |
|           |                                                          |
|           | **Fixed**                                                |
|           |                                                          |
|           | -   For function geam avoid overflow in offset           |
|           |     calculation.                                         |
|           |                                                          |
|           | -   For function syr avoid overflow in offset            |
|           |     calculation.                                         |
|           |                                                          |
|           | -   For function gemv (Transpose-case) avoid overflow in |
|           |     offset calculation.                                  |
|           |                                                          |
|           | -   For functions ssyrk and dsyrk, allow                 |
|           |     conjugate-transpose case to match legacy BLAS.       |
|           |     Behavior is the same as the transpose case.          |
+-----------+----------------------------------------------------------+
| hipBLAS   | **Added**                                                |
|           |                                                          |
|           | -   More support for hipblas-bench                       |
|           |                                                          |
|           | **Fixed**                                                |
|           |                                                          |
|           | -   Avoid large offset overflow for gemv and hemv in     |
|           |     hipblas-test                                         |
|           |                                                          |
|           | **Changed**                                              |
|           |                                                          |
|           | -   Packaging split into a runtime package called        |
|           |     hipblas and a development package called             |
|           |     hipblas-devel. The development package depends on    |
|           |     runtime. The runtime package suggests the            |
|           |     development package for all supported OSes except    |
|           |     CentOS 7 to aid in the transition. The suggests      |
|           |     feature in packaging is introduced as a deprecated   |
|           |     feature and will be removed in a future rocm         |
|           |     release.                                             |
+-----------+----------------------------------------------------------+
| rocFFT    | **Optimizations**                                        |
|           |                                                          |
|           | -   Optimized SBCC kernels of length 52, 60, 72, 80, 84, |
|           |     96, 104, 108, 112, 160, 168, 208, 216, 224, 240 with |
|           |     new kernel generator.                                |
|           |                                                          |
|           | **Added**                                                |
|           |                                                          |
|           | -   Split 2D device code into separate libraries.        |
|           |                                                          |
|           | **Changed**                                              |
|           |                                                          |
|           | -   Packaging split into a runtime package called rocfft |
|           |     and a development package called rocfft-devel. The   |
|           |     development package depends on runtime. The runtime  |
|           |     package suggests the development package for all     |
|           |     supported OSes except CentOS 7 to aid in the         |
|           |     transition. The suggests feature in packaging is     |
|           |     introduced as a deprecated feature and will be       |
|           |     removed in a future rocm release.                    |
|           |                                                          |
|           | **Fixed**                                                |
|           |                                                          |
|           | -   Fixed a few validation failures of even-length R2C   |
|           |     inplace. 2D, 3D cubics sizes such as 100\^2 (or      |
|           |     \^3), 200\^2 (or \^3), 256\^2 (or \^3)\...etc. We    |
|           |     don\'t combine the three kernels                     |
|           |     (stockham-r2c-transpose). We only combine two        |
|           |     kernels (r2c-transpose) instead.                     |
+-----------+----------------------------------------------------------+
| hipFFT    | **Changed**                                              |
|           |                                                          |
|           | -   Packaging split into a runtime package called hipfft |
|           |     and a development package called hipfft-devel. The   |
|           |     development package depends on runtime. The runtime  |
|           |     package suggests the development package for all     |
|           |     supported OSes except CentOS 7 to aid in the         |
|           |     transition. The suggests feature in packaging is     |
|           |     introduced as a deprecated feature and will be       |
|           |     removed in a future rocm release.                    |
+-----------+----------------------------------------------------------+
| rocSPARSE | **Added**                                                |
|           |                                                          |
|           | -   Triangular solve for multiple right-hand sides using |
|           |     BSR format                                           |
|           |                                                          |
|           | -   SpMV for BSRX format                                 |
|           |                                                          |
|           | -   SpMM in CSR format enhanced to work with transposed  |
|           |     A                                                    |
|           |                                                          |
|           | -   Matrix coloring for CSR matrices                     |
|           |                                                          |
|           | -   Added batched tridiagonal solve (gtsv_strided_batch) |
|           |                                                          |
|           | **Improved**                                             |
|           |                                                          |
|           | -   Fixed a bug with gemvi on Navi21                     |
|           |                                                          |
|           | -   Optimization for pivot based gtsv                    |
+-----------+----------------------------------------------------------+
| hipSPARSE | **Added**                                                |
|           |                                                          |
|           | -   Triangular solve for multiple right-hand sides using |
|           |     BSR format                                           |
|           |                                                          |
|           | -   SpMV for BSRX format                                 |
|           |                                                          |
|           | -   SpMM in CSR format enhanced to work with transposed  |
|           |     A                                                    |
|           |                                                          |
|           | -   Matrix coloring for CSR matrices                     |
|           |                                                          |
|           | -   Added batched tridiagonal solve (gtsv_strided_batch) |
|           |                                                          |
|           | **Improved**                                             |
|           |                                                          |
|           | -   Fixed a bug with gemvi on Navi21                     |
|           |                                                          |
|           | -   Optimization for pivot based gtsv                    |
+-----------+----------------------------------------------------------+
| r         | **Changed**                                              |
| ocALUTION |                                                          |
|           | -   Packaging split into a runtime package called        |
|           |     rocalution and a development package called          |
|           |     rocalution-devel. The development package depends on |
|           |     runtime. The runtime package suggests the            |
|           |     development package for all supported OSes except    |
|           |     CentOS 7 to aid in the transition. The suggests      |
|           |     feature in packaging is introduced as a deprecated   |
|           |     feature and will be removed in a future rocm         |
|           |     release.                                             |
|           |                                                          |
|           | **Improved**                                             |
|           |                                                          |
|           | -   (A)MG solving phase optimization                     |
+-----------+----------------------------------------------------------+
| rocTHRUST | **Changed**                                              |
|           |                                                          |
|           | -   Packaging changed to a development package (called   |
|           |     rocthrust-dev for .deb packages, and rocthrust-devel |
|           |     for .rpm packages). As rocThrust is a header-only    |
|           |     library, there is no runtime package. To aid in the  |
|           |     transition, the development package sets the         |
|           |     \"provides\" field to provide the package rocthrust, |
|           |     so that existing packages depending on rocthrust can |
|           |     continue to work. This provides feature is           |
|           |     introduced as a deprecated feature and will be       |
|           |     removed in a future ROCm release.                    |
+-----------+----------------------------------------------------------+
| rocSOLVER | **Added**                                                |
|           |                                                          |
|           | -   RQ factorization routines:                           |
|           |                                                          |
|           | -   GERQ2, GERQF (with batched and strided_batched       |
|           |     versions)                                            |
|           |                                                          |
|           | -   Linear solvers for general square systems:           |
|           |                                                          |
|           | -   GESV (with batched and strided_batched versions)     |
|           |                                                          |
|           | -   Linear solvers for symmetric/hermitian positive      |
|           |     definite systems:                                    |
|           |                                                          |
|           | -   POTRS (with batched and strided_batched versions)    |
|           |                                                          |
|           | -   POSV (with batched and strided_batched versions)     |
|           |                                                          |
|           | -   Inverse of symmetric/hermitian positive definite     |
|           |     matrices:                                            |
|           |                                                          |
|           | -   POTRI (with batched and strided_batched versions)    |
|           |                                                          |
|           | -   General matrix inversion without pivoting:           |
|           |                                                          |
|           | -   GETRI_NPVT (with batched and strided_batched         |
|           |     versions)                                            |
|           |                                                          |
|           | -   GETRI_NPVT_OUTOFPLACE (with batched and              |
|           |     strided_batched versions)                            |
|           |                                                          |
|           | **Optimized**                                            |
|           |                                                          |
|           | -   Improved performance of LU factorization (especially |
|           |     for large matrix sizes)                              |
|           |                                                          |
|           | -   Changed                                              |
|           |                                                          |
|           | -   Raised reference LAPACK version used for rocSOLVER   |
|           |     test and benchmark clients to v3.9.1                 |
|           |                                                          |
|           | -   Minor CMake improvements for users building from     |
|           |     source without install.sh:                           |
|           |                                                          |
|           | -   Removed fmt::fmt from rocsolver\'s public usage      |
|           |     requirements                                         |
|           |                                                          |
|           | -   Enabled small-size optimizations by default          |
|           |                                                          |
|           | -   Split packaging into a runtime package               |
|           |     (\'rocsolver\') and a development package            |
|           |     (\'rocsolver-devel\'). The development package       |
|           |     depends on the runtime package. To aid in the        |
|           |     transition, the runtime package suggests the         |
|           |     development package (except on CentOS 7). This use   |
|           |     of the suggests feature is deprecated and will be    |
|           |     removed in a future ROCm release.                    |
|           |                                                          |
|           | **Fixed**                                                |
|           |                                                          |
|           | -   Use of the GCC / Clang                               |
|           |     \_\_attribute\_\_((deprecated(\...))) extension is   |
|           |     now guarded by compiler detection macros.            |
+-----------+----------------------------------------------------------+
| hipSOLVER | The following functions were added in this release:      |
|           |                                                          |
|           | -   gesv                                                 |
|           |                                                          |
|           |     -   hipsolverSSgesv_bufferSize,                      |
|           |         hipsolverDDgesv_bufferSize,                      |
|           |         hipsolverCCgesv_bufferSize,                      |
|           |         hipsolverZZgesv_bufferSize                       |
|           |                                                          |
|           |     -   hipsolverSSgesv, hipsolverDDgesv,                |
|           |         hipsolverCCgesv, hipsolverZZgesv                 |
|           |                                                          |
|           | -   potrs                                                |
|           |                                                          |
|           |     -   hipsolverSpotrs_bufferSize,                      |
|           |         hipsolverDpotrs_bufferSize,                      |
|           |         hipsolverCpotrs_bufferSize,                      |
|           |         hipsolverZpotrs_bufferSize                       |
|           |                                                          |
|           |     -   hipsolverSpotrs, hipsolverDpotrs,                |
|           |         hipsolverCpotrs, hipsolverZpotrs                 |
|           |                                                          |
|           | -   potrsBatched                                         |
|           |                                                          |
|           |     -   hipsolverSpotrsBatched_bufferSize,               |
|           |         hipsolverDpotrsBatched_bufferSize,               |
|           |         hipsolverCpotrsBatched_bufferSize,               |
|           |         hipsolverZpotrsBatched_bufferSize                |
|           |                                                          |
|           |     -   hipsolverSpotrsBatched, hipsolverDpotrsBatched,  |
|           |         hipsolverCpotrsBatched, hipsolverZpotrsBatched   |
|           |                                                          |
|           | -   potri                                                |
|           |                                                          |
|           |     -   hipsolverSpotri_bufferSize,                      |
|           |         hipsolverDpotri_bufferSize,                      |
|           |         hipsolverCpotri_bufferSize,                      |
|           |         hipsolverZpotri_bufferSize                       |
|           |                                                          |
|           |     -   hipsolverSpotri, hipsolverDpotri,                |
|           |         hipsolverCpotri, hipsolverZpotri                 |
+-----------+----------------------------------------------------------+
| RCCL      | **Added**                                                |
|           |                                                          |
|           | -   Compatibility with NCCL 2.9.9                        |
|           |                                                          |
|           | **Changed**                                              |
|           |                                                          |
|           | -   Packaging split into a runtime package called rccl   |
|           |     and a development package called rccl-devel. The     |
|           |     development package depends on runtime. The runtime  |
|           |     package suggests the development package for all     |
|           |     supported OSes except CentOS 7 to aid in the         |
|           |     transition. The suggests feature in packaging is     |
|           |     introduced as a deprecated feature and will be       |
|           |     removed in a future rocm release.                    |
+-----------+----------------------------------------------------------+
| hipCUB    | **Changed**                                              |
|           |                                                          |
|           | -   Packaging changed to a development package (called   |
|           |     hipcub-dev for .deb packages, and hipcub-devel for   |
|           |     .rpm packages). As hipCUB is a header-only library,  |
|           |     there is no runtime package. To aid in the           |
|           |     transition, the development package sets the         |
|           |     \"provides\" field to provide the package hipcub, so |
|           |     that existing packages depending on hipcub can       |
|           |     continue to work. This provides feature is           |
|           |     introduced as a deprecated feature and will be       |
|           |     removed in a future ROCm release.                    |
+-----------+----------------------------------------------------------+
| rocPRIM   | **Added**                                                |
|           |                                                          |
|           | -   bfloat16 support added.                              |
|           |                                                          |
|           | **Changed**                                              |
|           |                                                          |
|           | -   Packaging split into a runtime package called        |
|           |     rocprim and a development package called             |
|           |     rocprim-devel. The development package depends on    |
|           |     runtime. The runtime package suggests the            |
|           |     development package for all supported OSes except    |
|           |     CentOS 7 to aid in the transition. The suggests      |
|           |     feature in packaging is introduced as a deprecated   |
|           |     feature and will be removed in a future rocm         |
|           |     release.                                             |
|           |                                                          |
|           | -   As rocPRIM is a header-only library, the runtime     |
|           |     package is an empty placeholder used to aid in the   |
|           |     transition. This package is also a deprecated        |
|           |     feature and will be removed in a future rocm         |
|           |     release.                                             |
|           |                                                          |
|           | **Deprecated**                                           |
|           |                                                          |
|           | -   The warp_size() function is now deprecated; please   |
|           |     switch to host_warp_size() and device_warp_size()    |
|           |     for host and device references respectively.         |
+-----------+----------------------------------------------------------+
| rocRAND   | **Changed**                                              |
|           |                                                          |
|           | -   Packaging split into a runtime package called        |
|           |     rocrand and a development package called             |
|           |     rocrand-devel. The development package depends on    |
|           |     runtime. The runtime package suggests the            |
|           |     development package for all supported OSes except    |
|           |     CentOS 7 to aid in the transition. The suggests      |
|           |     feature in packaging is introduced as a deprecated   |
|           |     feature and will be removed in a future rocm         |
|           |     release.                                             |
|           |                                                          |
|           | **Fixed**                                                |
|           |                                                          |
|           | -   Fix for mrg_uniform_distribution_double generating   |
|           |     incorrect range of values                            |
|           |                                                          |
|           | -   Fix for order of state calls for log_normal, normal, |
|           |     and uniform                                          |
|           |                                                          |
|           | **Known issues**                                         |
|           |                                                          |
|           | -   kernel_xorwow test is failing for certain GPU        |
|           |     architectures.                                       |
+-----------+----------------------------------------------------------+

For more information about ROCm Libraries, refer to the documentation at

<https://rocmdocs.amd.com/en/latest/ROCm_Libraries/ROCm_Libraries.html>

# Known Issues in This Release 

The following are the known issues in this release.

## Compiler Support for Function Pointers and Virtual Functions

A known issue in the compiler support for function pointers and virtual
functions on the GPU may cause undefined behavior due to register
corruption. 

A temporary workaround is to compile the affected application with *the
-mllvm -amdgpu-fixed-function-abi=1* option. 

**Note:** This is an internal compiler flag and may be removed without
notice once the issue is addressed in a future release.

## Debugger Process Exit May Cause ROCgdb Internal Error

If the debugger process exits during debugging, ROCgdb may report
internal errors. This issue occurs as it attempts to access the AMD GPU
state for the exited process. To recover, users must restart ROCgdb.

As a workaround, users can set breakpoints to prevent the debugged
process from exiting. For example, users can set breakpoints at the last
statement of the main function and in the abort() and exit() functions.
This temporary solution allows the application to be re-run without
restarting ROCgdb.

This issue is currently under investigation and will be fixed in a
future release.

For more information, refer to the ROCgdb User Guide at,

*Add link*

# Deprecations

## AMD Instinct MI25 End of Life

ROCm release v4.5 is the final release to support AMD Instinct MI25. AMD
Instinct MI25 has reached End of Life (EOL). ROCm 4.5 represents the
last certified release for software and driver support. AMD will
continue to provide technical support and issue resolution for AMD
Instinct MI25 on ROCm v4.5 for a period of 12 months from the software
GA date. 

# Hardware and Software Support

## Hardware Support

ROCm is focused on using AMD GPUs to accelerate computational tasks such
as machine learning, engineering workloads, and scientific computing. To
focus our development efforts on these domains of interest, ROCm
supports the following targeted set of hardware configurations.

### Supported Graphics Processing Units 

As the AMD ROCm platform has a focus on specific computational domains,
AMD offers official support for a selection of GPUs that are designed to
offer good performance and price in these domains.

**Note:** The integrated GPUs of Ryzen are not officially supported
targets for ROCm.

ROCm officially supports AMD GPUs that use the following chips:

-   GFX9 GPUs

```{=html}
<!-- -->
```
-   \"Vega 10\" chips, such as on the AMD Radeon RX Vega 64 and Radeon
    Instinct MI25

-   \"Vega 7nm\" chips, such as on the Radeon Instinct MI50, Radeon
    Instinct MI60, AMD Radeon VII, Radeon Pro VII

```{=html}
<!-- -->
```
-   CDNA GPUs

    -   MI100 chips such as on the AMD Instinct™ MI100

ROCm is a collection of software ranging from drivers and runtimes to
libraries and developer tools. Some of this software may work with more
GPUs than the \"officially supported\" list above, though AMD does not
make any official claims of support for these devices on the ROCm
software platform.

The following list of GPUs is enabled in the ROCm software. However,
full support is not guaranteed:

-   GFX8 GPUs

    -   \"Polaris 11\" chips, such as on the AMD Radeon RX 570 and
        Radeon Pro WX 4100

    -   \"Polaris 12\" chips, such as on the AMD Radeon RX 550 and
        Radeon RX 540

```{=html}
<!-- -->
```
-   

```{=html}
<!-- -->
```
-   GFX7 GPUs

    -   \"Hawaii\" chips, such as the AMD Radeon R9 390X and FirePro
        W9100

As described in the next section, GFX8 GPUs require PCI Express 3.0
(PCIe 3.0) with support for PCIe atomics. This requires both CPU and
motherboard support. GFX9 GPUs require PCIe 3.0 with support for PCIe
atomics by default, but they can operate in most cases without this
capability.

The integrated GPUs in AMD APUs are not officially supported targets for
ROCm. As
described [[below]{.underline}](https://github.com/Rmalavally/ROCm/blob/master/README.md#limited-support),
\"Carrizo\", \"Bristol Ridge\", and \"Raven Ridge\" APUs are enabled in
AMD upstream drivers and the ROCm OpenCL runtime. However, they are not
enabled in the HIP runtime, and may not work due to motherboard or OEM
hardware limitations. Note, they are not yet officially supported
targets for ROCm.

#### GFX8 GPUs­

**Note**: The GPUs require a host CPU and platform with PCIe 3.0 with
support for PCIe atomics.

+--------------------+---------------+---------------+---------------+
| GFX8 GPUs          |               |               |               |
+====================+===============+===============+===============+
| Fiji\              | **Polaris 10\ | **Polaris 11\ | **Polaris 12  |
| (AMD)              | (AMD)**       | (AMD)**       | (Lexa)\       |
|                    |               |               | (AMD)**       |
+--------------------+---------------+---------------+---------------+
| -   Radeon R9 Fury | -   Radeon RX | -   Radeon RX | -   Radeon RX |
|                    |     470       |     460       |     540       |
| -   Radeon R9 Nano |               |               |               |
|                    | -   Radeon RX | -   Radeon RX | -   Radeon RX |
| -   Radeon R9 Fury |     480       |     560       |     550       |
|     X              |               |               |               |
|                    | -   Radeon RX | -   Radeon    | -   Radeon    |
| -   Radeon Pro Duo |     570       |     Pro WX    |     Pro WX    |
|     (Fiji)         |               |     4100      |     2100      |
|                    | -   Radeon RX |               |               |
| -   FirePro S9300  |     580       |               | -   Radeon    |
|     X2             |               |               |     Pro WX    |
|                    | -   Radeon    |               |     3100      |
| -   Radeon         |     Pro Duo   |               |               |
|     Instinct MI8   |     (Polaris) |               |               |
|                    |               |               |               |
|                    | -   Radeon    |               |               |
|                    |     Pro WX    |               |               |
|                    |     5100      |               |               |
|                    |               |               |               |
|                    | -   Radeon    |               |               |
|                    |     Pro WX    |               |               |
|                    |     7100      |               |               |
|                    |               |               |               |
|                    | -   Radeon    |               |               |
|                    |     Instinct  |               |               |
|                    |     MI6       |               |               |
+--------------------+---------------+---------------+---------------+

#### GFX9 GPUs

ROCm offers support for two chips from AMD's most recent "gfx9"
generation of GPUs.

+-----------------------------------+-----------------------------------+
| GFX9 GPUs                         |                                   |
+===================================+===================================+
| Vega 10\                          | **Vega 7nm\                       |
| (AMD)                             | (AMD)**                           |
+-----------------------------------+-----------------------------------+
| -   Radeon RX Vega 56             | -   Radeon VII                    |
|                                   |                                   |
| -   Radeon RX Vega 64             | -   Radeon Instinct MI50          |
|                                   |                                   |
| -   Radeon Vega Frontier Edition  | -   Radeon Instinct MI60          |
|                                   |                                   |
| -   Radeon Pro WX 8200            | ```{=html}                        |
|                                   | <!-- -->                          |
| -   Radeon Pro WX 9100            | ```                               |
|                                   | -                                 |
| -   Radeon Pro V340               |                                   |
|                                   |                                   |
| -   Radeon Pro V340 MxGPU         |                                   |
|                                   |                                   |
| -   Radeon Instinct MI25          |                                   |
|                                   |                                   |
| Note: ROCm does not support       |                                   |
| Radeon Pro SSG.                   |                                   |
+-----------------------------------+-----------------------------------+

-   

## Supported CPUs

As described above, GFX8 GPUs require PCIe 3.0 with PCIe atomics to run
ROCm. In particular, the CPU and every active PCIe point between the CPU
and GPU require support for PCIe 3.0 and PCIe atomics. The CPU root must
indicate PCIe AtomicOp Completion capabilities and any intermediate
switch must indicate PCIe AtomicOp Routing capabilities.

The current CPUs which support PCIe Gen3 + PCIe Atomics are:

-   AMD Ryzen CPUs

-   CPUs in AMD Ryzen APUs

-   AMD Ryzen Threadripper CPUs

-   AMD EPYC CPUs

-   Intel Xeon E7 v3 or newer CPUs

-   Intel Xeon E5 v3 or newer CPUs

-   Intel Xeon E3 v3 or newer CPUs

-   Intel Core i7 v4, Core i5 v4, Core i3 v4 or newer CPUs (i.e. Haswell
    family or newer)

-   Some Ivy Bridge-E systems

Beginning with ROCm 1.8, GFX9 GPUs (such as Vega 10) no longer require
PCIe atomics. We have similarly made more options available for many
PCIe lanes. GFX9 GPUs can now be run on CPUs without PCIe atomics and on
older PCIe generations, such as PCIe 2.0. This is not supported on GPUs
below GFX9, e.g. GFX8 cards in the Fiji and Polaris families.

If you are using any PCIe switches in your system, please note that PCIe
Atomics are only supported on some switches, such as Broadcom PLX. When
you install your GPUs, make sure you install them in a PCIe 3.0 x16, x8,
x4, or x1 slot attached either directly to the CPU\'s Root I/O
controller or via a PCIe switch directly attached to the CPU\'s Root I/O
controller.

In our experience, many issues stem from trying to use consumer
motherboards which provide physical x16 connectors that are electrically
connected as e.g. PCIe 2.0 x4, PCIe slots connected via the Southbridge
PCIe I/O controller, or PCIe slots connected through a PCIe switch that
does not support PCIe atomics.

If you attempt to run ROCm on a system without proper PCIe atomic
support, you may see an error in the kernel log (dmesg):

*kfd: skipped device 1002:7300, PCI rejects atomics*

Experimental support for our Hawaii (GFX7) GPUs (Radeon R9 290, R9 390,
FirePro W9100, S9150, S9170) does not require or take advantage of PCIe
Atomics. However, AMD recommends that you use a CPU from the list
provided above for compatibility purposes.

## Not supported or limited support under ROCm

##### Limited support

-   ROCm 4.x should support PCIe 2.0 enabled CPUs such as the AMD
    Opteron, Phenom, Phenom II, Athlon, Athlon X2, Athlon II and older
    Intel Xeon and Intel Core Architecture and Pentium CPUs. However, we
    have done very limited testing on these configurations, since our
    test farm has been catering to CPUs listed above. This is where we
    need community support. 

```{=html}
<!-- -->
```
-   Please report these issues.

```{=html}
<!-- -->
```
-   Thunderbolt 1, 2, and 3 enabled breakout boxes should now be able to
    work with ROCm. Thunderbolt 1 and 2 are PCIe 2.0 based, and thus are
    only supported with GPUs that do not require PCIe 3.0 atomics (e.g.
    Vega 10). However, we have done no testing on this configuration and
    would need community support due to limited access to this type of
    equipment.

-   AMD \"Carrizo\" and \"Bristol Ridge\" APUs are enabled to run
    OpenCL, but do not yet support HIP or our libraries built on top of
    these compilers and runtimes.

    -   As of ROCm 2.1, \"Carrizo\" and \"Bristol Ridge\" require the
        use of upstream kernel drivers.

    -   In addition, various \"Carrizo\" and \"Bristol Ridge\" platforms
        may not work due to OEM and ODM choices when it comes to key
        configuration parameters such as the inclusion of the required
        CRAT tables and IOMMU configuration parameters in the system
        BIOS.

    -   Before purchasing such a system for ROCm, please verify that the
        BIOS provides an option for enabling IOMMUv2 and that the system
        BIOS properly exposes the correct CRAT table. Inquire with your
        vendor about the latter.

-   AMD \"Raven Ridge\" APUs are enabled to run OpenCL, but do not yet
    support HIP or our libraries built on top of these compilers and
    runtimes.

    -   As of ROCm 2.1, \"Raven Ridge\" requires the use of upstream
        kernel drivers.

    -   In addition, various \"Raven Ridge\" platforms may not work due
        to OEM and ODM choices when it comes to key configuration
        parameters such as the inclusion of the required CRAT tables and
        IOMMU configuration parameters in the system BIOS.

    -   Before purchasing such a system for ROCm, please verify that the
        BIOS provides an option for enabling IOMMUv2 and that the system
        BIOS properly exposes the correct CRAT table. Inquire with your
        vendor about the latter.

##### Not supported

-   \"Tonga\", \"Iceland\", \"Vega M\", and \"Vega 12\" GPUs are not
    supported.

-   AMD does not support GFX8-class GPUs (Fiji, Polaris, etc.) on CPUs
    that do not have PCIe3.0 with PCIe atomics.

    -   AMD Carrizo and Kaveri APUs as hosts for such GPUs are not
        supported

    -   Thunderbolt 1 and 2 enabled GPUs are not supported by GFX8 GPUs
        on ROCm. Thunderbolt 1 & 2 are based on PCIe 2.0.

In the default ROCm configuration, GFX8 and GFX9 GPUs require PCI
Express 3.0 with PCIe atomics. The ROCm platform leverages these
advanced capabilities to allow features such as user-level submission of
work from the host to the GPU. This includes PCIe atomic Fetch and Add,
Compare and Swap, Unconditional Swap, and AtomicOp Completion.

Current CPUs which support PCIe 3.0 + PCIe Atomics:

+-----------------------------------+-----------------------------------+
| AMD                               | INTEL                             |
+===================================+===================================+
| Ryzen CPUs (Family 17h Model      | Intel Core i3, i5, and i7 CPUs    |
| 01h-0Fh)                          | from Haswell and beyond. This     |
|                                   | includes:                         |
| -   Ryzen 3 1300X                 |                                   |
|                                   | -   Haswell CPUs such as the Core |
| -   Ryzen 3 2300X                 |     i7 4790K                      |
|                                   |                                   |
| -   Ryzen 5 1600X                 | -   Broadwell CPUs such as the    |
|                                   |     Core i7 5775C                 |
| -   Ryzen 5 2600X                 |                                   |
|                                   | -   Skylake CPUs such as the Core |
| -   Ryzen 7 1800X                 |     i7 6700K                      |
|                                   |                                   |
| -   Ryzen 7 2700X                 | -   Kaby Lake CPUs such as the    |
|                                   |     Core i7 7740X                 |
|                                   |                                   |
|                                   | -   Coffee Lake CPUs such as the  |
|                                   |     Core i7 8700K                 |
|                                   |                                   |
|                                   | -   Xeon CPUs from "v3" and newer |
|                                   |                                   |
|                                   | -   Some models of "Ivy Bridge-E" |
|                                   |     processors                    |
+-----------------------------------+-----------------------------------+
| Ryzen APUs (Family 17h Model      |                                   |
| 10h-1Fh -- previously code-named  |                                   |
| Raven Ridge) such as:             |                                   |
|                                   |                                   |
| -   Athlon 200GE                  |                                   |
|                                   |                                   |
| -   Ryzen 5 2400G                 |                                   |
|                                   |                                   |
| Note: The integrated GPU in these |                                   |
| devices is not guaranteed to work |                                   |
| with ROCm.                        |                                   |
+-----------------------------------+-----------------------------------+
| Ryzen Threadripper Workstation    |                                   |
| CPUs (Family 17h Model 01h-0Fh)   |                                   |
| such as:                          |                                   |
|                                   |                                   |
| -   Ryzen Threadripper 1950X      |                                   |
|                                   |                                   |
| -   Ryzen Threadripper 2990WX     |                                   |
+-----------------------------------+-----------------------------------+
| EPYC Server CPUs (Family 17h      |                                   |
| Model 01h-0Fh) such as:           |                                   |
|                                   |                                   |
| -   Epyc 7551P                    |                                   |
|                                   |                                   |
| -   Epyc 7601                     |                                   |
+-----------------------------------+-----------------------------------+

## ROCm Support in Upstream Linux Kernels

As of ROCm 1.9.0, the ROCm user-level software is compatible with the
AMD drivers in certain upstream Linux kernels.

As such, users have the option of either using the ROCK kernel driver
that are part of AMD\'s ROCm repositories or using the upstream driver
and only installing ROCm user-level utilities from AMD\'s ROCm
repositories.

These releases of the upstream Linux kernel support the following GPUs
in ROCm:

-   4.17: Fiji, Polaris 10, Polaris 11

-   4.18: Fiji, Polaris 10, Polaris 11, Vega10

-   4.20: Fiji, Polaris 10, Polaris 11, Vega10, Vega 7nm

The upstream driver may be useful for running ROCm software on systems
that are not compatible with the kernel driver available in AMD\'s
repositories.

For users that have the option of using either AMD\'s or the upstreamed
driver, there are various tradeoffs to take into consideration:

  ------------------------------------------------------------------------
             Using AMD\'s rock-dkms   Using the upstream kernel driver
             package                  
  ---------- ------------------------ ------------------------------------
                                      

  Pros       More GPU features and    Includes the latest Linux kernel
             are enabled earlier      features

             Tested by AMD on         May work on other distributions and
             supported distributions  with custom kernels

             Supported GPUs enabled   
             regardless of kernel     
             version                  

             Includes the latest GPU  
             firmware                 

  Cons       May not work on all      Features and hardware support varies
             Linux distributions or   depending on the kernel version
             versions                 

             Not currently supported  Limits GPU\'s usage of system memory
             on kernels newer than    to 3/8 of system memory (before
             5.4                      5.6). For 5.6 and beyond, both DKMS
                                      and upstream kernels allow the use
                                      of 15/16 of system memory.

                                      IPC and RDMA capabilities are not
                                      yet enabled

                                      Not tested by AMD to the same level
                                      as rock-dkms package

                                      Does not include the most up-to-date
                                      firmware
  ------------------------------------------------------------------------

# 

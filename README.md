# ROCm™ Repository Updates
This repository contains the manifest file for ROCm™ releases, changelogs, and release information. The file default.xml contains information for all repositories and the associated commit used to build the current ROCm release.

The default.xml file uses the repo Manifest format.

# ROCm v5.4 Release Notes
ROCm v5.4 is now released. For ROCm v5.4 documentation, refer to https://docs.amd.com.

# ROCm v5.3.3 Release Notes
ROCm v5.3.3 is now released. For ROCm v5.3.3 documentation, refer to https://docs.amd.com.

# ROCm v5.3.2 Release Notes
ROCm v5.3.2 is now released. For ROCm v5.3.2 documentation, refer to https://docs.amd.com.

# ROCm v5.3 Release Notes
ROCm v5.3 is now released. For ROCm v5.3 documentation, refer to https://docs.amd.com.

# ROCm v5.2.3 Release Notes
The ROCm v5.2.3 patch release is now available. The details are listed below. Highlights of this release include enhancements in RCCL version compatibility and minor bug fixes in the HIP Runtime.

Additionally, ROCm releases will return to use of the [ROCm](https://github.com/RadeonOpenCompute/ROCm) repository for version-controlled release notes henceforth.

**NOTE**: This release of ROCm is validated with the AMDGPU release v22.20.1.

All users of the ROCm v5.2.1 release and below are encouraged to upgrade. Refer to https://docs.amd.com for documentation associated with this release.


## Introducing Preview Support for Ubuntu 20.04.5 HWE

Refer to the following article for information on the preview support for Ubuntu 20.04.5 HWE.

https://www.amd.com/en/support/kb/release-notes/rn-amdgpu-unified-linux-22-20


## Changes in This Release

### Ubuntu 18.04 End of Life

Support for Ubuntu 18.04 ends in this release. Future releases of ROCm will not provide prebuilt packages for Ubuntu 18.04.


### HIP and Other Runtimes

#### HIP Runtime

##### Fixes

 - A bug was discovered in the HIP graph capture implementation in the ROCm v5.2.0 release. If the same kernel is called twice (with different argument values) in a graph capture, the implementation only kept the argument values for the second kernel call.

- A bug was introduced in the hiprtc implementation in the ROCm v5.2.0 release. This bug caused the *hiprtcGetLoweredName* call to fail for named expressions with whitespace in it.

**Example:** The named expression ```my_sqrt<complex<double>>``` passed but ```my_sqrt<complex<double>>``` failed.


### ROCm Libraries

#### RCCL

##### Added
- Compatibility with NCCL 2.12.10
- Packages for test and benchmark executables on all supported OSes using CPack
- Adding custom signal handler - opt-in with RCCL_ENABLE_SIGNALHANDLER=1
  - Additional details provided if Binary File Descriptor library (BFD) is pre-installed.
  - Adding experimental support for using multiple ranks per device
    - Requires using a new interface to create communicator (ncclCommInitRankMulti),
        refer to the interface documentation for details.
	  - To avoid potential deadlocks, user might have to set an environment variables increasing
	      the number of hardware queues. For example,

For more information, see https://docs.amd.com.



# AMD ROCm™ v5.0 Release Notes


# ROCm Installation Updates

This document describes the features, fixed issues, and information about downloading and installing the AMD ROCm™ software.

It also covers known issues and deprecations in this release.

## Notice for Open-source and Closed-source ROCm Repositories in Future Releases

To make a distinction between open-source and closed-source components, all ROCm repositories will consist of sub-folders in future releases.

- All open-source components will be placed in the `base-url/<rocm-ver>/main` sub-folder
- All closed-source components will reside in the `base-url/<rocm-ver>/proprietary`  sub-folder

## List of Supported Operating Systems

The AMD ROCm platform supports the following operating systems:

| **OS-Version (64-bit)** | **Kernel Versions** |
| --- | --- |
| CentOS 8.3 | 4.18.0-193.el8 |
| CentOS 7.9 | 3.10.0-1127 |
| RHEL 8.5 | 4.18.0-348.7.1.el8\_5.x86\_64 |
| RHEL 8.4 | 4.18.0-305.el8.x86\_64 |
| RHEL 7.9 | 3.10.0-1160.6.1.el7 |
| SLES 15 SP3 | 5.3.18-59.16-default |
| Ubuntu 20.04.3 | 5.8.0 LTS / 5.11 HWE |
| Ubuntu 18.04.5 [5.4 HWE kernel] | 5.4.0-71-generic |

### Support for RHEL v8.5

This release extends support for RHEL v8.5.

### Supported GPUs - Unofficial -
This document is a work in progress. It is a work in progress in a feature branch based on community feedback. 


Support can be assumed for current server accelerators and GPUs intended for enterprise customers.
This ROCm release officially supports workstation GPU products.
However, there are also compatible consumer GPU products available.
Finally, this ROCm release can be used with a number of older GPU products.

#### Enterprise GPUs

Current enterprise GPU accelerators can be assumed to be supported. These include:

- AMD Instinct™ MI100
- AMD Instinct™ MI210
- AMD Instinct™ MI250
- AMD Instinct™ MI250X

#### Supported Workstation GPUs

This ROCm release extends support to two workstation GPU products:

- AMD Radeon™ PRO V620
- AMD Radeon™ PRO W6800

These features are verified to work by quality assurance (QA):

- SRIOV virtualization support for AMD Radeon™ PRO V620
- KVM Hypervisor (1VF support only) on Ubuntu Host OS with Ubuntu, CentOs, and RHEL Guest
- Support for ROCm-SMI in an SRIOV environment. For more details, refer to [the ROCm SMI API documentation](https://rocmdocs.amd.com/en/latest/ROCm_System_Managment/ROCm-System-Managment.html).

**Note:** AMD Radeon™ PRO V620 is not supported on SLES.

#### Compatible Consumer GPUs

All the supported workstation GPU products use the `gfx1030` instruction set architecture (ISA) implemented in the "Navi 21" silicon.
The same silicon type implementing the same ISA is also used in some consumer GPU products, which *should* therefore be compatible with this ROCm release.
However, ROCm was not validated against the following products before release:

-   AMD Radeon™ RX 6800
-   AMD Radeon™ RX 6800 XT
-   AMD Radeon™ RX 6900 XT
-   AMD Radeon™ RX 6950 XT

#### Library Target Matrix

In general, the libraries of this ROCm release can be used with GPU products which implement one of the instruction set architectures (ISA) listed in each of the libraries `TARGETS` variables.
The current target GPU products with relevant instruction set architectures are:

| ISA           | Family     | Chip            | Product                                           | RCCL | rocBLAS | rocFFT | rocPRIM | rocRAND | rocSolver | rocSparse | rocThrust | rocALUTION | rocWMMA |
|:--------------|:-----------|:----------------|:--------------------------------------------------|:----:|:-------:|:------:|:-------:|:-------:|:---------:|:---------:|:---------:|:----------:|:-------:|
| gfx803        | Fiji       | Fiji XT         | AMD Radeon™ R9 Fury                               |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx803        | Fiji       | Fiji XT         | AMD Radeon™ R9 Nano                               |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx803        | Fiji       | Fiji XT         | AMD Radeon™ R9 Fury X                             |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx803        | Fiji       | Fiji XT         | AMD Radeon Instinct™ MI8                         |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx803        | Fiji       | Capsaicin XT    | AMD FirePro™  S9300x2                             |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx803        | Fiji       | Capsaicin XT    | AMD Radeon™ PRO Duo 2016                          |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx803        | Polaris    | Polaris 11      | AMD Radeon™ PRO WX 4100                           |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx803        | Polaris    | Polaris 11      | AMD Radeon™ PRO WX 4130 Mobile                    |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx803        | Polaris    | Polaris 11      | AMD Radeon™ PRO WX 4150 Mobile                    |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx803        | Polaris    | Polaris 11      | AMD Radeon™ PRO WX 4170 Mobile                    |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx803        | Polaris    | Polaris 11      | AMD Radeon™ RX 460                                |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx803        | Polaris    | Polaris 11      | AMD Radeon™ RX 560D                               |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx803        | Polaris    | Polaris 10      | AMD Radeon Instinct™ MI6                         |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx803        | Polaris    | Polaris 10      | AMD Radeon™ PRO Duo 2017                          |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx803        | Polaris    | Polaris 10      | AMD Radeon™ PRO WX 5100                           |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx803        | Polaris    | Polaris 10      | AMD Radeon™ PRO WX 7100                           |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx803        | Polaris    | Polaris 10      | AMD Radeon™ PRO WX 7100 Mobile                    |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx803        | Polaris    | Polaris 10      | AMD Radeon™ RX 470                                |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx803        | Polaris    | Polaris 10      | AMD Radeon™ RX 480                                |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx803        | Polaris    | Polaris 21      | AMD Radeon™ PRO 555                               |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx803        | Polaris    | Polaris 21      | AMD Radeon™ PRO 555X                              |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx803        | Polaris    | Polaris 21      | AMD Radeon™ PRO 560                               |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx803        | Polaris    | Polaris 21      | AMD Radeon™ PRO 560X                              |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx803        | Polaris    | Polaris 20      | AMD Radeon™ PRO 575                               |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx803        | Polaris    | Polaris 20      | AMD Radeon™ RX 580                                |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx803        | Polaris    | Polaris 20      | AMD Radeon™ PRO 580                               |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx803        | Polaris    | Polaris 20      | AMD Radeon™ RX 570                                |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx803        | Polaris    | Polaris 20      | AMD Radeon™ PRO 570                               |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx803        | Polaris    | Polaris 30      | AMD Radeon™ RX 590                                |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx900        | Vega       | Vega 10 XL      | AMD Radeon™ RX Vega 56                            |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx900        | Vega       | Vega 10 XL      | AMD Radeon™ PRO Vega 56                           |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx900        | Vega       | Vega 10 XT      | AMD Radeon Instinct™ MI25                        |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx900        | Vega       | Vega 10 XT      | AMD Radeon™ RX Vega 64                            |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx900        | Vega       | Vega 10 XT      | AMD Radeon™ RX Vega 64 Liquid                     |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx900        | Vega       | Vega 10 XTX AIR | AMD Radeon™ Vega Frontier Edition                 |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx900        | Vega       | Vega 10 XTX LCS | AMD Radeon™ Vega Frontier Edition (Liquid Cooled) |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx906        | Radeon VII | Vega 20 GL      | AMD Radeon Instinct™ MI50                        |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx906        | Radeon VII | Vega 20 GL      | AMD Radeon Instinct™ MI60                        |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx906        | Radeon VII | Vega 20 XT      | AMD Radeon™ VII                                   |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx906        | Radeon VII | Vega 20 XT      | AMD Radeon™ PRO VII                               |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx908        | MI100      | MI100 XL        | AMD Instinct™ MI100                               |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ✅    |
| gfx90a        | Aldebaran  | MI200           | AMD Instinct™ MI210                               |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ✅    |
| gfx90a        | Aldebaran  | MI200           | AMD Instinct™ MI250                               |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ✅    |
| gfx90a        | Aldebaran  | MI200           | AMD Instinct™ MI250X                              |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ✅    |
| gfx1010       | Navi       | Navi 10 XT      | AMD Radeon™ PRO 5600M                             |  ❌   |    ✅    |   ❌    |    ❌    |    ❌    |     ✅     |     ❌     |     ❌     |     ❌      |    ∅    |
| gfx1010       | Navi       | Navi 10 XLE     | AMD Radeon™ PRO 5600 XT                           |  ❌   |    ✅    |   ❌    |    ❌    |    ❌    |     ✅     |     ❌     |     ❌     |     ❌      |    ∅    |
| gfx1010       | Navi       | Navi 10 XL      | AMD Radeon™ RX 5700                               |  ❌   |    ✅    |   ❌    |    ❌    |    ❌    |     ✅     |     ❌     |     ❌     |     ❌      |    ∅    |
| gfx1010       | Navi       | Navi 10 XT      | AMD Radeon™ RX 5700 XT                            |  ❌   |    ✅    |   ❌    |    ❌    |    ❌    |     ✅     |     ❌     |     ❌     |     ❌      |    ∅    |
| gfx1012       | Navi       | Navi 14 XT      | AMD Radeon™ RX 5500                               |  ❌   |    ✅    |   ❌    |    ❌    |    ❌    |     ✅     |     ❌     |     ❌     |     ❌      |    ∅    |
| gfx1012       | Navi       | Navi 14 XTX     | AMD Radeon™ RX 5500 XT                            |  ❌   |    ✅    |   ❌    |    ❌    |    ❌    |     ✅     |     ❌     |     ❌     |     ❌      |    ∅    |
| gfx1030       | Navi 2X    | Navi 21         | AMD Radeon™ RX 6800                               |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx1030       | Navi 2X    | Navi 21         | AMD Radeon™ RX 6800 XT                            |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx1030       | Navi 2X    | Navi 21         | AMD Radeon™ RX 6900 XT                            |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx1030       | Navi 2X    | Navi 21         | AMD Radeon™ RX 6950 XT                            |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx1030       | Navi 2X    | Navi 21         | AMD Radeon™ PRO V620                              |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |
| gfx1030       | Navi 2X    | Navi 21         | AMD Radeon™ PRO W6800                             |  ✅   |    ✅    |   ✅    |    ✅    |    ✅    |     ✅     |     ✅     |     ✅     |     ✅      |    ∅    |

Legend of this table:

| Symbol |           Meaning           |
|:------:|:---------------------------:|
|   ∅    |    No required hardware     |
|   ✅    |     Library targets GPU     |
|   ❌    | Library omits targeting GPU |

Sources of this table:

- [`TARGETS` in RCCL's CMakeLists.txt](https://github.com/ROCmSoftwarePlatform/rccl/blob/develop/CMakeLists.txt#L35)
- [`TARGETS` in rocBLAS' CMakeLists.txt](https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/develop/CMakeLists.txt#L117)
- [`TARGETS` in rocFFT's CMakeLists.txt](https://github.com/ROCmSoftwarePlatform/rocFFT/blob/develop/CMakeLists.txt#L155)
- [`TARGETS` in rocPRIM's CMakeLists.txt](https://github.com/ROCmSoftwarePlatform/rocPRIM/blob/develop/CMakeLists.txt#L74)
- [`TARGETS` in rocRAND's CMakeLists.txt](https://github.com/ROCmSoftwarePlatform/rocRAND/blob/develop/CMakeLists.txt#L82)
- [`TARGETS` in rocSolver's CMakeLists.txt](https://github.com/ROCmSoftwarePlatform/rocSOLVER/blob/develop/CMakeLists.txt#L142)
- [`TARGETS` in rocSparse's CMakeLists.txt](https://github.com/ROCmSoftwarePlatform/rocSPARSE/blob/develop/CMakeLists.txt#L153)
- [`TARGETS` in rocWMMA's CMakeLists.txt](https://github.com/ROCmSoftwarePlatform/rocWMMA/blob/develop/CMakeLists.txt#L77)
- [`TARGETS` in rocThrust's CMakeLists.txt](https://github.com/ROCmSoftwarePlatform/rocThrust/blob/develop/CMakeLists.txt#L62)
- [`TARGETS` in rocALUTION's CMakeLists.txt](https://github.com/ROCmSoftwarePlatform/rocALUTION/blob/develop/CMakeLists.txt#L82)

> Note: A ROCm library being able to run on a given GPU product does not mean that it works correctly on said hardware.


## Older ROCm™ Releases
For release information for older ROCm™ releases, refer to [CHANGELOG](CHANGELOG.md).

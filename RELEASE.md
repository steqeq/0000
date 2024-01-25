# Release notes
<!-- Disable lints since this is an auto-generated file.    -->
<!-- markdownlint-disable blanks-around-headers             -->
<!-- markdownlint-disable no-duplicate-header               -->
<!-- markdownlint-disable no-blanks-blockquote              -->
<!-- markdownlint-disable ul-indent                         -->
<!-- markdownlint-disable no-trailing-spaces                -->

<!-- spellcheck-disable -->

This page contains the release notes for AMD ROCm Software.

-------------------

## ROCm 6.0.2

In addition to library updates for several ROCm projects, this release includes bug fixes and
improvements for the following libraries: AMDMIGraphX, hipBLASLt, hipFFT, hipRAND, hipSPARSELt,
rocSPARSE, rocThrust, rocWMMA, and Tensile. You can find additional details in the following section.

### Library changes in ROCM 6.0.2

| Library | Version |
|---------|---------|
| AMDMIGraphX |  ⇒ [2.8](https://github.com/ROCm/AMDMIGraphX/releases/tag/rocm-6.0.2) |
| hipBLAS |  ⇒ [2.0.0](https://github.com/ROCm/hipBLAS/releases/tag/rocm-6.0.2) |
| hipBLASLt |  ⇒ [0.6.0](https://github.com/ROCm/hipBLASLt/releases/tag/rocm-6.0.2) |
| hipCUB |  ⇒ [3.0.0](https://github.com/ROCm/hipCUB/releases/tag/rocm-6.0.2) |
| hipFFT |  ⇒ [1.0.13](https://github.com/ROCm/hipFFT/releases/tag/rocm-6.0.2) |
| hipRAND |  ⇒ [2.10.17](https://github.com/ROCm/hipRAND/releases/tag/rocm-6.0.2) |
| hipSOLVER |  ⇒ [2.0.0](https://github.com/ROCm/hipSOLVER/releases/tag/rocm-6.0.2) |
| hipSPARSE |  ⇒ [3.0.0](https://github.com/ROCm/hipSPARSE/releases/tag/rocm-6.0.2) |
| hipSPARSELt |  ⇒ [0.1.0](https://github.com/ROCm/hipSPARSELt/releases/tag/rocm-6.0.2) |
| hipTensor |  ⇒ [1.1.0](https://github.com/ROCm/hipTensor/releases/tag/rocm-6.0.2) |
| MIOpen |  ⇒ [2.19.0](https://github.com/ROCm/MIOpen/releases/tag/rocm-6.0.2) |
| rccl |  ⇒ [2.15.5](https://github.com/ROCm/rccl/releases/tag/rocm-6.0.2) |
| rocALUTION |  ⇒ [3.0.3](https://github.com/ROCm/rocALUTION/releases/tag/rocm-6.0.2) |
| rocBLAS |  ⇒ [4.0.0](https://github.com/ROCm/rocBLAS/releases/tag/rocm-6.0.2) |
| rocFFT |  ⇒ [1.0.25](https://github.com/ROCm/rocFFT/releases/tag/rocm-6.0.2) |
| rocm-cmake |  ⇒ [0.11.0](https://github.com/ROCm/rocm-cmake/releases/tag/rocm-6.0.2) |
| rocPRIM |  ⇒ [3.0.0](https://github.com/ROCm/rocPRIM/releases/tag/rocm-6.0.2) |
| rocRAND |  ⇒ [3.0.0](https://github.com/ROCm/rocRAND/releases/tag/rocm-6.0.2) |
| rocSOLVER |  ⇒ [3.24.0](https://github.com/ROCm/rocSOLVER/releases/tag/rocm-6.0.2) |
| rocSPARSE |  ⇒ [3.0.2](https://github.com/ROCm/rocSPARSE/releases/tag/rocm-6.0.2) |
| rocThrust |  ⇒ [3.0.0](https://github.com/ROCm/rocThrust/releases/tag/rocm-6.0.2) |
| rocWMMA |  ⇒ [1.3.0](https://github.com/ROCm/rocWMMA/releases/tag/rocm-6.0.2) |
| Tensile |  ⇒ [4.39.0](https://github.com/ROCm/Tensile/releases/tag/rocm-6.0.2) |

#### AMDMIGraphX 2.8

MIGraphX 2.8 for ROCm 6.0.2

##### Additions

* Support for MI300 GPUs

#### hipBLASLt 0.6.0

hipBLASLt 0.6.0 for ROCm 6.0.2

##### Additions

* Added `UserArguments` for `GroupedGemm`
* Support for datatype: FP16 in with FP32 out
* New samples
* Support for datatype: `Int8` in `Int32` out
* Support for gfx94x platform
* Support for FP8/BF8 datatype (only for gfx94x platform)
* Support Scalar A,B,C,D for FP8/BF8 datatype

##### Changes

* Replaced `hipblasDatatype_t` with `hipDataType`
* Replaced `hipblasLtComputeType_t` with `hipblasComputeType_t`
* Deprecated `HIPBLASLT_MATMUL_DESC_D_SCALE_VECTOR_POINTER`

#### hipFFT 1.0.13

hipFFT 1.0.13 for ROCm 6.0.2

##### Changes

* Removed the Git submodule for shared files between rocFFT and hipFFT; instead, just copy the files
 over (this should help simplify downstream builds and packaging)

#### hipRAND 2.10.17

hipRAND 2.10.17 for ROCm 6.0.2

##### Fixes

* Fixed benchmark and unit test builds on Windows

#### hipSPARSELt 0.1.0

hipSPARSELt 0.1.0 for ROCm 6.0.2

##### Additions

* Enabled hipSPARSELt APIs
* Support for:
  * gfx940, gfx941, and gfx942 platforms
  * FP16, BF16, and INT8 problem types
  * ReLU, GELU, abs, sigmod, tanh activation
  * GELU scaling
  * Bias vector
  * Batched computation (single sparse x multiple dense, multiple sparse x single dense)
  * cuSPARSELt v0.4 backend
* Integration with TensileLite kernel generator
* GoogleTest: hipsparselt-test
* Benchmarking tool: hipsparselt-bench
* Sample app: `example_spmm_strided_batched`, `example_prune, example_compress`

#### rocSPARSE 3.0.2

rocSPARSE 3.0.2 for ROCm 6.0.2

##### Optimizations

* Optimization to doti routine
* Optimization to spin-looping algorithms

##### Deprecations

* `rocsparse_spmv_ex`
* `rocsparse_xbsrmv_ex routines`

#### rocThrust 3.0.0

rocThrust 3.0.0 for ROCm 6.0.2

##### Removals

* Removed CUB symlink from the root of the repository
* Removed support for deprecated macros (`THRUST_DEVICE_BACKEND` and
  `THRUST_HOST_BACKEND`)

##### Fixes

* Fixed a segmentation fault when binary search/upper bound/lower bound/equal range was invoked
  with the `hip_rocprim::execute_on_stream_base` policy

##### Known issues

* Some branches in Thrust's code may be unreachable in the HIP backend
  * With the NVIDIA backend, `NV_IF_TARGET` and `THRUST_RDC_ENABLED` are used as a substitute for
    the `THRUST_HAS_CUDART` macro, which is now no longer used in Thrust (legacy support only)
  * There is no `THRUST_RDC_ENABLED` macro available for the HIP backend

#### rocWMMA 1.3.0

rocWMMA 1.3.0 for ROCm 6.0.2

##### Additions

* Support for:
  * gfx940, gfx941, and gfx942 targets

#### Tensile 4.39.0

Tensile 4.39.0 for ROCm 6.0.2

##### Optimizations

* Adjusted `miIssueLatency` for gfx940

##### Changes

* Enabled ROCm SMI for gfx940/941
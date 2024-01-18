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

## ROCm 6.0.1

In addition to library updates for several ROCm projects, this release includes bug fixes and
improvements for the following projects: hipBLAS, hipBLASLt, hipSPARSELt, rocAL, and rocBLAS. You
can find additional details in the following section.

### Library changes in ROCM 6.0.1

| Library | Version |
|---------|---------|
| AMDMIGraphX | [2.8](https://github.com/ROCm/AMDMIGraphX/releases/tag/rocm-6.0.1) |
| hipBLAS | [2.0.0](https://github.com/ROCm/hipBLAS/releases/tag/rocm-6.0.1) |
| hipCUB | [3.0.0](https://github.com/ROCm/hipCUB/releases/tag/rocm-6.0.1) |
| hipFFT | [1.0.13](https://github.com/ROCm/hipFFT/releases/tag/rocm-6.0.1) |
| hipRAND | [2.10.17](https://github.com/ROCm/hipRAND/releases/tag/rocm-6.0.1) |
| hipSOLVER | [2.0.0](https://github.com/ROCm/hipSOLVER/releases/tag/rocm-6.0.1) |
| hipSPARSE | [3.0.0](https://github.com/ROCm/hipSPARSE/releases/tag/rocm-6.0.1) |
| hipSPARSELt |  ⇒ [0.1.0](https://github.com/ROCm/hipSPARSELt/releases/tag/rocm-6.0.1) |
| hipTensor | [1.1.0](https://github.com/ROCm/hipTensor/releases/tag/rocm-6.0.1) |
| MIOpen | [2.19.0](https://github.com/ROCm/MIOpen/releases/tag/rocm-6.0.1) |
| rccl | [2.15.5](https://github.com/ROCm/rccl/releases/tag/rocm-6.0.1) |
| rocALUTION | [3.0.3](https://github.com/ROCm/rocALUTION/releases/tag/rocm-6.0.1) |
| rocBLAS | [4.0.0](https://github.com/ROCm/rocBLAS/releases/tag/rocm-6.0.1) |
| rocFFT | [1.0.25](https://github.com/ROCm/rocFFT/releases/tag/rocm-6.0.1) |
| rocm-cmake | [0.11.0](https://github.com/ROCm/rocm-cmake/releases/tag/rocm-6.0.1) |
| rocPRIM | [3.0.0](https://github.com/ROCm/rocPRIM/releases/tag/rocm-6.0.1) |
| rocRAND | [2.10.17](https://github.com/ROCm/rocRAND/releases/tag/rocm-6.0.1) |
| rocSOLVER | [3.24.0](https://github.com/ROCm/rocSOLVER/releases/tag/rocm-6.0.1) |
| rocSPARSE | [3.0.2](https://github.com/ROCm/rocSPARSE/releases/tag/rocm-6.0.1) |
| rocThrust | [3.0.0](https://github.com/ROCm/rocThrust/releases/tag/rocm-6.0.1) |
| rocWMMA | [1.3.0](https://github.com/ROCm/rocWMMA/releases/tag/rocm-6.0.1) |
| Tensile | [4.39.0](https://github.com/ROCm/Tensile/releases/tag/rocm-6.0.1) |

#### hipBLAS 2.1.0

hipBLAS 2.1.0 for ROCm 6.0.1

#### Changes

* Some Level 2 function argument names have changed from `m` to `n` to match legacy BLAS; there
  was no change in implementation.
* Updated client code to use YAML-based testing
* Renamed `.doxygen` and `.sphinx` folders to `doxygen` and `sphinx`, respectively
* Added CMake support for documentation

#### hipBLASLt 0.7.0

hipBLASLt 0.7.0 for ROCm 6.0.1

##### Additions

* Extension APIs:
  * `hipblasltExtSoftmax`
  * `hipblasltExtLayerNorm`
  * `hipblasltExtAMax`
* `GemmTuning` extension parameter to set split-k by user
* Support for mixed-precision datatype: FP16/FP8 in with FP16 out

#### hipSPARSELt 0.1.0

hipSPARSELt 0.1.0 for ROCm 6.0.1

##### Additions

* Enabled hipSPARSELt APIs
* Support for:
  * gfx940, gfx941, and gfx942 platforms
  * FP16, BF16, and INT8 problem types
  * ReLU, GELU, abs, sigmod, and tanh activation
  * GELU scaling
  * Bias vectors
  * cuSPARSELt v0.4 backend
* Integrated with Tensile Lite kernel generator
* Support for batched computation (single sparse x multiple dense and multiple sparse x
single dense)
* GoogleTest: hipsparselt-test
* `hipsparselt-bench` benchmarking tool
* Sample apps: `example_spmm_strided_batched`, `example_prune`, `example_compress`

#### rocAL 1.0.0

rocAL 1.0.0 for ROCm 6.0.1

##### Changes

* Removed CuPy from `setup.py`

#### rocBLAS 4.1.0

rocBLAS 4.1.0 for ROCm 6.0.1

##### Additions

* Level 1 and Level 1 Extension functions have additional ILP64 API for both C and Fortran (`_64` name
  suffix) with int64_t function arguments
* Cache flush timing for `gemm_ex`

##### Changes

* Some Level 2 function argument names have changed `m` to `n` to match legacy BLAS; there is no
  change in implementation
* Standardized the use of non-blocking streams for copying results from device to host

##### Fixes

* Fixed host-pointer mode reductions for non-blocking streams

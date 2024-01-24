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
improvements for the following libraries: hipBLAS, hipBLASLt, hipSPARSELt, rocAL, and rocBLAS. You
can find additional details in the following section.

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
* Support for TorchMIGraphX via PyTorch
* Boosted overall performance by integrating rocMLIR
* INT8 support for ONNX Runtime
* Support for ONNX version 1.14.1
* Added new operators: `Qlinearadd`, `QlinearGlobalAveragePool`, `Qlinearconv`, `Shrink`, `CastLike`,
  and `RandomUniform`
* Added an error message for when `gpu_targets` is not set during MIGraphX compilation
* Added parameter to set tolerances with `migraphx-driver` verify
* Added support for MXR files &gt; 4 GB
* Added `MIGRAPHX_TRACE_MLIR` flag
* BETA added capability for using ROCm Composable Kernels via the `MIGRAPHX_ENABLE_CK=1`
  environment variable

##### Optimizations

* Improved performance support for INT8
* Improved time precision while benchmarking candidate kernels from CK or MLIR
* Removed contiguous from reshape parsing
* Updated the `ConstantOfShape` operator to support Dynamic Batch
* Simplified dynamic shapes-related operators to their static versions, where possible
* Improved debugging tools for accuracy issues
* Included a print warning about `miopen_fusion` while generating `mxr`
* General reduction in system memory usage during model compilation
* Created additional fusion opportunities during model compilation
* Improved debugging for matchers
* Improved general debug messages

##### Fixes

* Fixed scatter operator for nonstandard shapes with some models from ONNX Model Zoo
* Provided a compile option to improve the accuracy of some models by disabling Fast-Math
* Improved layernorm + pointwise fusion matching to ignore argument order
* Fixed accuracy issue with `ROIAlign` operator
* Fixed computation logic for the `Trilu` operator
* Fixed support for the DETR model

##### Changes

* Changed MIGraphX version to 2.8
* Extracted the test packages into a separate deb file when building MIGraphX from source

##### Removals

* Removed building Python 2.7 bindings

#### hipBLAS 2.0.0

hipBLAS 2.0.0 for ROCm 6.0.2

##### Added

* Option to define `HIPBLAS_USE_HIP_BFLOAT16` to switch API to use `hip_bfloat16` type
* `hipblasGemmExWithFlags` API

##### Deprecations

* `hipblasDatatype_t` is deprecated and will be removed in a future release; use `hipDataType` instead
* `hipblasComplex` is deprecated and will be removed in a future release; use `hipComplex` instead
* `hipblasDoubleComplex`  is deprecated and will be removed in a future release; use
  `hipDoubleComplex` instead
* Using `hipblasDatatype_t` for `hipblasGemmEx` for compute-type is deprecated and will be replaced
  with `hipblasComputeType_t` in a future release

##### Removals

* `hipblasXtrmm` (which calculates `B &lt;- alpha * op(A) * B`) has been replaced with `hipblasXtrmm`
  (which calculates `C &lt;- alpha * op(A) * B`)

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

#### hipCUB 3.0.0

hipCUB 3.0.0 for ROCm 6.0.2

##### Changes

* Removed `DOWNLOAD_ROCPRIM`
  * You can force rocPRIM to download using `DEPENDENCIES_FORCE_DOWNLOAD`

#### hipFFT 1.0.13

hipFFT 1.0.13 for ROCm 6.0.2

##### Changes

* Removed the Git submodule for shared files between rocFFT and hipFFT; instead, just copy the files
 over (this should help simplify downstream builds and packaging)

#### hipRAND 2.10.17

hipRAND 2.10.17 for ROCm 6.0.2

##### Fixes

* Fixed benchmark and unit test builds on Windows

#### hipSOLVER 2.0.0

hipSOLVER 2.0.0 for ROCm 6.0.2

##### Additions

* hipBLAS is now an optional dependency to `hipsolver-test`
  * Use the `BUILD_HIPBLAS_TESTS` CMake option to test compatibility between hipSOLVER and
    hipBLAS

##### Changes

* Types `hipsolverOperation_t`, `hipsolverFillMode_t`, and `hipsolverSideMode_t` are now aliases of
  `hipblasOperation_t`, `hipblasFillMode_t`, and `hipblasSideMode_t`

##### Fixes

* Fixed tests for hipSOLVER info updates in `ORGBR/UNGBR`, `ORGQR/UNGQR`,
  `ORGTR/UNGTR`, `ORMQR/UNMQR`, and `ORMTR/UNMTR`

#### hipSPARSE 3.0.0

hipSPARSE 3.0.0 for ROCm 6.0.2

##### Additions

* `hipsparseGetErrorName` and `hipsparseGetErrorString`

##### Changes

* Changed `hipsparseSpSV_solve()` API function to match cuSPARSE API
* Changed generic API functions to use `const` descriptors
* Improved documentation

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

#### hipTensor 1.1.0

hipTensor 1.1.0 for ROCm 6.0.2

##### Additions

* Architecture support for gfx940, gfx941, and gfx942
* Client tests configuration parameters now support YAML file input format

##### Changes

* Doxygen now treats warnings as errors

##### Fixes

* Client tests output redirections now behave accordingly
* Removed dependency static library deployment
* Security issues for documentation
* Compile issues in debug mode
* Corrected soft link for ROCm deployment

#### MIOpen 2.19.0

MIOpen 2.19.0 for ROCm 6.0.2

##### Additions

* ROCm 5.5 support for gfx1101 (Navi32)

##### Changes

* Tuning results for MLIR on ROCm 5.5
* Bumping MLIR commit to 5.5.0 release tag

##### Fixes

* 3D convolution Host API bug
* [HOTFIX][MI200][FP16] Disabled `ConvHipImplicitGemmBwdXdlops` when `FP16_ALT` is required

#### rccl 2.15.5

RCCL 2.15.5 for ROCm 6.0.2

##### Changes

* Compatibility with NCCL 2.15.5
* Unit test executable renamed to `rccl-UnitTests`

##### Additions

* HW Topology-aware binary tree implementation
* Experimental support for MSCCL
* New unit tests for hipGraph support
* NPKit integration

##### Fixes

* `rocm-smi` ID conversion
* Support for `HIP_VISIBLE_DEVICES` for unit tests
* Support for p2p transfers to non (HIP) visible devices

##### Removals

* Removed TransferBench from tools
  * It now exists in a standalone repository: https://github.com/ROCmSoftwarePlatform/TransferBench

#### rocALUTION 3.0.3

rocALUTION 3.0.3 for ROCm 6.0.2

##### Additions

* Support for 64bit integer vectors
* Inclusive and exclusive sum functionality for vector classes
* Transpose functionality for Global/LocalMatrix
* TripleMatrixProduct functionality LocalMatrix
* `Sort()` function for LocalVector class
* Multiple stream support to the HIP backend

##### Optimizations

* `GlobalMatrix::Apply()` now uses multiple streams to better hide communication

##### Changes

* Matrix dimensions and number of non-zeros are now stored using 64-bit integers
* Improved ILUT preconditioner

##### Removals

* `LocalVector::GetIndexValues(ValueType\*)`
* `LocalVector::SetIndexValues(const ValueType\*)`
* `LocalMatrix::RSDirectInterpolation(const LocalVector&amp;, const LocalVector&amp;, LocalMatrix\*, LocalMatrix\*)`
* `LocalMatrix::RSExtPIInterpolation(const LocalVector&amp;, const LocalVector&amp;, bool, float, LocalMatrix\*, LocalMatrix\*)`
* `LocalMatrix::RugeStueben()`
* `LocalMatrix::AMGSmoothedAggregation(ValueType, const LocalVector&amp;, const LocalVector&amp;, LocalMatrix\*, LocalMatrix\*, int)`
* `LocalMatrix::AMGAggregation(const LocalVector&amp;, LocalMatrix\*, LocalMatrix\*)`

##### Fixes

* Unit tests no longer ignore the BCSR block dimension
* Typos in the documentation
* Bug in multi-coloring for non-symmetric matrix patterns

#### rocBLAS 4.0.0

rocBLAS 4.0.0 for ROCm 6.0.2

##### Additions

* Beta API `rocblas_gemm_batched_ex3` and `rocblas_gemm_strided_batched_ex3`
* Added input/output type `f16_r`/`bf16_r` and execution type `f32_r` support for Level 2
  `gemv_batched` and `gemv_strided_batched`
* `rocblas_status_excluded_from_build` will be used when calling functions that require Tensile (when
  using rocBLAS built without Tensile)
* System for async kernel launches setting a failure `rocblas_status` based on `hipPeekAtLastError`
  discrepancy

##### Optimizations

* Trsm performance for small sizes m &lt; 32 &amp;&amp; n &lt; 32

##### Deprecations

* In a future release, atomic operations will be disabled by default (to allow for repeatable results)
  * You can always enable (or disable) atomic operations using the `rocblas_set_atomics_mode` function
  * Enabling atomic operations can improve performance

##### Removals

* `rocblas_gemm_ext2` API function
* In-place trmm API from Legacy BLAS
  * This is replaced with an API that supports both in-place and out-of-place trmm
* INT8x4 support (INT8 support is unchanged)
* The #define `STDC_WANT_IEC_60559_TYPES_EXT` has been removed from `rocblas-types.h`
  * If you want ISO/IEC TS 18661-3:2015 functionality, you must define
    `STDC_WANT_IEC_60559_TYPES_EXT` before including `float.h`, `math.h`, and `rocblas.h`
* The default build removes device code for gfx803 architecture from the fat binary

##### Fixes

* Offset calculations for rocBLAS functions are now 64-bit safe
  * Fixes for very large leading dimension or increment potentially causing overflow:
    * Level2: `gbmv`, `gemv`, `hbmv`, `sbmv`, `spmv`, `tbmv`, `tpmv`, `tbsv`, `tpsv`
* Lazy loading supports a heterogeneous architecture setup and loads the appropriate Tensile library
  files based on the device's architecture
* Guard against no-op kernel launches resulting in potential `hipGetLastError`

##### Changes

* Reduced the default verbosity of `rocblas-test`
  * To see all tests, set the `GTEST_LISTENER=PASS_LINE_IN_LOG` environment variable

#### rocFFT 1.0.25

rocFFT 1.0.25 for ROCm 6.0.2

##### Additions

* Implemented experimental APIs to allow computation of FFTs on data distributed across multiple
  devices in a single process.
  * `rocfft_field` is a new type you can add to a plan description to describe the layout of FFT input or
    output.
  * `rocfft_field_add_brick` can be called one or more times to describe a brick decomposition of an FFT
    field, where each brick can be assigned a different device

  ```{note}
  These interfaces are still experimental and subject to change. We are interested in feedback. You can
  raise questions and concerns by opening an issue in the
  [rocFFT issue tracker](https://github.com/ROCmSoftwarePlatform/rocFFT/issues).
  ```
  *At this time, multi-device FFTs have several limitations (which will be removed in future releases):
    * Real-complex (forward or inverse) FFTs are not currently supported
    * Planar format fields are not currently supported
    * Batch (i.e., `number_of_transforms` provided to `rocfft_plan_create`) must be 1
    * The FFT input is gathered to the current device at run time, so all FFT data must fit on that device


##### Optimizations

* Improved the performance of some small 2D/3D real FFTs supported by the `2D_SINGLE` kernel
  * gfx90a gets more optimization by offline tuning
* Removed an extra kernel launch from even-length, real-complex FFTs that use callbacks

##### Changes

* Built kernels in solution-map to library kernel cache
* Real forward transforms (real-to-complex) no longer overwrite input
  * rocFFT still may overwrite real inverse (complex-to-real) input, as this allows for faster performance
* `rocfft-rider` and `dyna-rocfft-rider` have been renamed to `rocfft-bench` and `dyna-rocfft-bench`
  * These are controlled by the `BUILD_CLIENTS_BENCH CMake` option.
  * Links for the old file names are installed, and the old `BUILD_CLIENTS_RIDER CMake` option is
    accepted for compatibility, but will be removed in a future release
* Binaries in debug builds no longer have a `-d` suffix

##### Fixes

* rocFFT now correctly handles load callbacks that convert data from a smaller data type (e.g., 16-bit integers -&gt; 32-bit float)

#### rocm-cmake 0.11.0

rocm-cmake 0.11.0 for ROCm 6.0.2

##### Changes

* Improved validation, documentation, and rocm-docs-core integration for ROCMSphinxDoc

##### Fixes

* ROCMClangTidy: Fixed extra `make` flags passed for Clang-Tidy
* ROCMTest: Fixed issues when using module in a subdirectory

#### rocPRIM 3.0.0

rocPRIM 3.0.0 for ROCm 6.0.2

##### Additions

* `block_sort::sort()` overload for keys and values with a dynamic size for all block sort algorithms
  * All `block_sort::sort()` overloads with a dynamic size are now supported for
    `block_sort_algorithm::merge_sort` and `block_sort_algorithm::bitonic_sort`
* New two-way partition primitive (`partition_two_way`) that can write to two separate iterators

##### Optimizations

* Improved the performance of `partition`

##### Fixes

* `rocprim::MatchAny` for devices with 64-bit warp size
  * The `rocprim::MatchAny` function has been deprecated and replaced with `rocprim::match_any`

#### rocRAND 3.0.0

rocRAND 3.0.0 for ROCm 6.0.2

##### Changes

* Generator classes from `rocrand.hpp` are no longer copyable
  * In previous versions, copies would copy internal references to the generators, leading to double free
    or memory leak errors
  * These types should be moved instead of copied, and move constructors and operators are now
    defined for them

##### Optimizations

* Improved MT19937 initialization and generation performance

##### Removals

* Removed hipRAND submodule from rocRAND: hipRAND is now only available as a separate package
* Removed references to, and workarounds for, the deprecated `hcc`

##### Fixes

* `mt19937_engine` from `rocrand.hpp` is now move-constructible and move-assignable
  * The move constructor and move assignment operator was previously deleted for this class
* Various fixes for the C++ wrapper header `rocrand.hpp`
  * Fixed the spelling of `mrg31k3p` (it was incorrectly named`mrg31k3a` in previous versions)
  * Added missing `order` setter method for `threefry4x64`
  * Fixed the default ordering parameter for `lfsr113`
* Build error when using Clang++ directly due to unsupported references to amdgpu-target

#### rocSOLVER 3.24.0

rocSOLVER 3.24.0 for ROCm 6.0.2

##### Additions

* Cholesky refactorization for sparse matrices: `CSRRF_REFACTCHOL`
* `rocsolver_rfinfo_mode`, and the ability to specify the desired refactorization routine (see
  `rocsolver_set_rfinfo_mode`)

##### Changes

* `CSRRF_ANALYSIS` and `CSRRF_SOLVE` now support sparse Cholesky factorization

#### rocSPARSE 3.0.2

rocSPARSE 3.0.2 for ROCm 6.0.2

##### Additions

* `rocsparse_inverse_permutation`
* Mixed precisions for SpVV
* Uniform INT8 precision for Gather and Scatter

##### Optimizations

* Optimization to doti routine
* Optimization to spin-looping algorithms

##### Changes

* Changed `rocsparse_spmv` function arguments
* Changed `rocsparse_xbsrmv` routines function arguments
* You now have to call `hipStreamSynchronize` after doti, dotci, spvv, and csr2ell when using host
  pointer mode
* Improved documentation
* Improved verbose output during argument checking on API function calls

##### Deprecatations

* `rocsparse_spmv_ex`
* `rocsparse_xbsrmv_ex routines`

##### Removals

* Auto stages from spmv, spmm, spgemm, spsv, spsm, and spitsv
* `rocsparse_spmm_ex` routine

##### Fixes

* Bug in `rocsparse-bench`, where the SpMV algorithm was not taken into account in CSR format
* BSR/GEBSR routines bsrmv, bsrsv, bsrmm, bsrgeam, gebsrmv, and gebsrmm, so that `block_dim==0`
  is considered an invalid size
* Bug where passing `nnz = 0` to doti or dotci did not always return a dot product of 0

#### rocThrust 3.0.0

rocThrust 3.0.0 for ROCm 6.0.2

##### Additions

* Updated to match upstream Thrust 2.0.1
* `NV_IF_TARGET` macro from libcu++ for NVIDIA backend and HIP implementation for HIP backend

##### Changes

* The CMake build system now additionally accepts `GPU_TARGETS` in addition to `AMDGPU_TARGETS`
  for setting the targeted GPU architectures
  * `GPU_TARGETS=all` compiles for all supported architectures
  * `AMDGPU_TARGETS` is only provided for backwards compatibility; `GPU_TARGETS` is preferred

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
  * gfx940, gfx941 and gfx942 targets
  * f8, BF8 and xfloat32 datatypes
  * `HIP_NO_HALF` and `__ HIP_NO_HALF_CONVERSIONS__ and __ HIP_NO_HALF_OPERATORS__` (e.g.,
    PyTorch environment)

##### Changes

* rocWMMA with hipRTC now supports `bfloat16_t` datatype
* gfx11 WMMA now uses lane swap instead of broadcast for layout adjustments
* Updated samples GEMM parameter validation on host architecture

##### Fixes

* Disabled GoogleTest static library deployment
* Extended tests now build in large code model

#### Tensile 4.39.0

Tensile 4.39.0 for ROCm 6.0.2

##### Additions

* Aqua Vanjaram support for:
  * gfx940, gfx941, gfx942
  * FP8, BF8, and XF32 datatypes
  * Stochastic rounding for various datatypes
* Added and updated tuning scripts
* `DirectToLds` support for larger data types with 32-bit global load, and the corresponding test cases
  * `DirectToLds` is being replaced with `DirectToLdsA` and `DirectToLdsB`
* Added the average of frequency, power consumption, and temperature for the winner kernels to the
  CSV file
* ASMCap check for MFMA + const src
* Support for wider local read + pack with `v_perm` (with `VgprForLocalReadPacking=True`)
* New parameter to increase `miLatencyLeft`

##### Optimizations

* Enabled `InitAccVgprOpt` for `MatrixInstruction` cases
* Implemented local read-related parameter calculations with `DirectToVgpr`
* Adjusted `miIssueLatency` for gfx940
* Enabled dedicated VGPR allocation for local read + pack
* Optimized code initialization
* Optimized SPGR allocation
* Support for DGEMM TLUB + RLVW=2 for odd N (edge shift change)
* Enabled `miLatency` optimization for (gfx940/gfx941 + MFMA) for specific data types, and fixed
  instruction scheduling

##### Changes

* Removed old code for DTL + (bpe * GlobalReadVectorWidth &gt; 4)
* Changed and updated failed CI tests for `gfx11xx`, `InitAccVgprOpt`, and `DTLds`
* Removed unused `CustomKernels` and `ReplacementKernels`
* Added a reject condition for DTVB + TransposeLDS=False (not supported so far)
* Removed unused code for `DirectToLds`
* Updated test cases for DTV + `TransposeLDS=False`
* Moved `MinKForGSU` parameter from `globalparameter` to `BenchmarkCommonParameter` in order
  to support smaller K
* Changed how to calculate `latencyForLR` for `miLatency`
* Set a minimum value of `latencyForLRCount` for `1LDSBuffer` to avoid being rejected by
  `overflowedResources=5` (related to miLatency)
* Refactored `allowLRVWBforTLUandMI` and renamed it to `VectorWidthB`
* Support for multi-GPU for different architectures in lazy library loading
* Enabled `dtree` library for batch &gt; 1
* Added problem scale feature for `dtree` selection
* Enabled ROCm SMI for gfx940/941
* Modified non-lazy load build to skip experimental logic

##### Fixes

* Predicate ordering for FP16alt impl round near zero mode to unbreak distance modes
* Boundary check for mirror dims and re-enabled the mirror dims test cases
* Merge error affecting i8 with WMMA
* Mismatch issue with DTLds + TSGR + TailLoop
* Bug with `InitAccVgprOpt` + GSU&gt;1, and a mismatch issue with PGR=0
* Override for unloaded solutions when lazy loading
* Build errors (added missing headers)
* Boost link for a clean build on Ubuntu 22
* Bug in `forcestoresc1` arch selection
* Compiler directive for gfx941 and gfx942
* Formatting for `DecisionTree_test.cpp`
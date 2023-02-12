# ROCm™ Repository Updates
This repository contains the manifest file for ROCm™ releases, changelogs, and release information. The file default.xml contains information for all repositories and the associated commit used to build the current ROCm release.

The default.xml file uses the repo Manifest format.

# ROCm v5.4.3 Release Notes
ROCm v5.4.3 is now released. For ROCm v5.4.3 documentation, refer to https://docs.amd.com.

# ROCm v5.4.2 Release Notes
ROCm v5.4.2 is now released. For ROCm v5.4.2 documentation, refer to https://docs.amd.com.

# ROCm v5.4.1 Release Notes
ROCm v5.4.1 is now released. For ROCm v5.4.1 documentation, refer to https://docs.amd.com.

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

#### Tensile

##### Added
- Async DMA support for Transpose Data Layout (ThreadSeparateGlobalReadA/B)
- Option to output library logic in dictionary format
- No solution found error message for benchmarking client
- Exact K check for StoreCInUnrollExact
- Support for CGEMM + MIArchVgpr
- client-path parameter for using prebuilt client
- CleanUpBuildFiles global parameter
- Debug flag for printing library logic index of winning solution
- NumWarmups global parameter for benchmarking
- Windows support for benchmarking client
- DirectToVgpr support for CGEMM
- TensileLibLogicToYaml for creating tuning configs from library logic solutions
##### Optimizations
- Put beta code and store separately if StoreCInUnroll = x4 store
- Improved performance for StoreCInUnroll + b128 store
##### Changed
- Re-enable HardwareMonitor for gfx90a
- Decision trees use MLFeatures instead of Properties
##### Fixed
- Reject DirectToVgpr + MatrixInstBM/BN > 1
- Fix benchmark timings when using warmups and/or validation
- Fix mismatch issue with DirectToVgprB + VectorWidth > 1
- Fix mismatch issue with DirectToLds + NumLoadsCoalesced > 1 + TailLoop
- Fix incorrect reject condition for DirectToVgpr
- Fix reject condition for DirectToVgpr + MIWaveTile < VectorWidth
- Fix incorrect instruction generation with StoreCInUnroll

#### hipBLAS

##### Added
- Allow for selection of int8 datatype
- Added support for hipblasXgels and hipblasXgelsStridedBatched operations (with s,d,c,z precisions),
  only supported with rocBLAS backend
- Added support for hipblasXgelsBatched operations (with s,d,c,z precisions)

#### hipCUB

##### Added
- CMake functionality to improve build parallelism of the test suite that splits compilation units by
function or by parameters.
- New overload for `BlockAdjacentDifference::SubtractLeftPartialTile` that takes a predecessor item.
##### Changed
- Improved build parallelism of the test suite by splitting up large compilation units for `DeviceRadixSort`,
`DeviceSegmentedRadixSort` and `DeviceSegmentedSort`.
- CUB backend references CUB and thrust version 1.17.1.

#### hipFFT


##### Added
- Added hipfftExtPlanScaleFactor API to efficiently multiply each output element of a FFT by a given scaling factor.  Result scaling must be supported in the backend FFT library.

##### Changed
- When hipFFT is built against the rocFFT backend, rocFFT 1.0.19 or higher is now required.

#### hipSOLVER

##### Added
- Added compatibility-only functions
  - gesvdaStridedBatched
    - hipsolverDnSgesvdaStridedBatched_bufferSize, hipsolverDnDgesvdaStridedBatched_bufferSize, hipsolverDnCgesvdaStridedBatched_bufferSize, hipsolverDnZgesvdaStridedBatched_bufferSize
    - hipsolverDnSgesvdaStridedBatched, hipsolverDnDgesvdaStridedBatched, hipsolverDnCgesvdaStridedBatched, hipsolverDnZgesvdaStridedBatched


#### hipSPARSE

##### Added
- Added hipsparseCsr2cscEx2_bufferSize and hipsparseCsr2cscEx2 routines
##### Changed
- HIPSPARSE_ORDER_COLUMN has been renamed to HIPSPARSE_ORDER_COL to match cusparse

#### rccl

##### Changed
- Compatibility with NCCL 2.13.4
- Improvements to RCCL when running with hipGraphs
- RCCL_ENABLE_HIPGRAPH environment variable is no longer necessary to enable hipGraph support
- Minor latency improvements
##### Fixed
- Resolved potential memory access error due to asynchronous memset

#### rocALUTION

##### Added
- Added build support for Navi31 and Navi33
- Added support for non-squared global matrices
##### Improved
- Fixed a memory leak in MatrixMult on HIP backend
- Global structures can now be used with a single process
##### Changed
- Switched GTest death test style to 'threadsafe'
- GlobalVector::GetGhostSize() is deprecated and will be removed
- ParallelManager::GetGlobalSize(), ParallelManager::GetLocalSize(), ParallelManager::SetGlobalSize() and ParallelManager::SetLocalSize() are deprecated and will be removed
- Vector::GetGhostSize() is deprecated and will be removed
- Multigrid::SetOperatorFormat(unsigned int) is deprecated and will be removed, use Multigrid::SetOperatorFormat(unsigned int, int) instead
- RugeStuebenAMG::SetCouplingStrength(ValueType) is deprecated and will be removed, use SetStrengthThreshold(float) instead

#### rocBLAS

##### Added
- client smoke test dataset added for quick validation using command rocblas-test --yaml rocblas_smoke.yaml
- Added stream order device memory allocation as a non-default beta option.

##### Optimized
- Improved trsm performance for small sizes by using a substitution method technique
- Improved syr2k and her2k performance significantly by using a block-recursive algorithm

##### Changed
- Level 2, Level 1, and Extension functions: argument checking when the handle is set to rocblas_pointer_mode_host now returns the status of rocblas_status_invalid_pointer only for pointers that must be dereferenced based on the alpha and beta argument values.  With handle mode rocblas_pointer_mode_device only pointers that are always dereferenced regardless of alpha and beta values are checked and so may lead to a return status of rocblas_status_invalid_pointer.   This improves consistency with legacy BLAS behaviour.
- Add variable to turn on/off ieee16/ieee32 tests for mixed precision gemm
- Allow hipBLAS to select int8 datatype
- Disallow B == C && ldb != ldc in rocblas_xtrmm_outofplace

##### Fixed
- FORTRAN interfaces generalized for FORTRAN compilers other than gfortran
- fix for trsm_strided_batched rocblas-bench performance gathering
- Fix for rocm-smi path in commandrunner.py script to match ROCm 5.2 and above

#### rocFFT


##### Optimizations
- Optimized some strided large 1D plans.

##### Added
- Added rocfft_plan_description_set_scale_factor API to efficiently multiply each output element of a FFT by a given scaling factor.
- Created a rocfft_kernel_cache.db file next to the installed library. SBCC kernels are moved to this file when built with the library, and are runtime-compiled for new GPU architectures.
- Added gfx1100 and gfx1102 to default AMDGPU_TARGETS.

##### Changed
- Moved runtime compilation cache to in-memory by default.  A default on-disk cache can encounter contention problems
on multi-node clusters with a shared filesystem.  rocFFT can still be told to use an on-disk cache by setting the
ROCFFT_RTC_CACHE_PATH environment variable.

#### rocPRIM

##### Changed
- `device_partition`, `device_unique`, and `device_reduce_by_key` now support problem
  sizes larger than 2^32 items.
##### Removed
- `block_sort::sort()` overload for keys and values with a dynamic size. This overload was documented but the
  implementation is missing. To avoid further confusion the documentation is removed until a decision is made on
  implementing the function.
##### Fixed
- Fixed the compilation failure in `device_merge` if the two key iterators don't match.

#### rocRAND

##### Added
- MRG31K3P pseudorandom number generator based on L'Ecuyer and Touzin, 2000, "Fast combined multiple recursive generators with multipliers of the form a = ±2q ±2r".
- LFSR113 pseudorandom number generator based on L'Ecuyer, 1999, "Tables of maximally equidistributed combined LFSR generators".
- SCRAMBLED_SOBOL32 and SCRAMBLED_SOBOL64 quasirandom number generators. The Scrambled Sobol sequences are generated by scrambling the output of a Sobol sequence.
##### Changed
- The `mrg_<distribution>_distribution` structures, which provided numbers based on MRG32K3A, are now replaced by `mrg_engine_<distribution>_distribution`, where `<distribution>` is `log_normal`, `normal`, `poisson`, or `uniform`. These structures provide numbers for MRG31K3P (with template type `rocrand_state_mrg31k3p`) and MRG32K3A (with template type `rocrand_state_mrg32k3a`).
##### Fixed
- Sobol64 now returns 64 bits random numbers, instead of 32 bits random numbers. As a result, the performance of this generator has regressed.
- Fixed a bug that prevented compiling code in C++ mode (with a host compiler) when it included the rocRAND headers on Windows.

#### rocSOLVER

##### Added
- Partial SVD for bidiagonal matrices:
    - BDSVDX
- Partial SVD for general matrices:
    - GESVDX (with batched and strided\_batched versions)

##### Changed
- Changed `ROCSOLVER_EMBED_FMT` default to `ON` for users building directly with CMake.
  This matches the existing default when building with install.sh or rmake.py.


#### rocSPARSE

##### Added
- Added rocsparse_spmv_ex routine
- Added rocsparse_bsrmv_ex_analysis and rocsparse_bsrmv_ex routines
- Added csritilu0 routine
- Added build support for Navi31 and Navi 33
##### Improved
- Optimization to segmented algorithm for COO SpMV by performing analysis
- Improve performance when generating random matrices.
- Fixed bug in ellmv
- Optimized bsr2csr routine
- Fixed integer overflow bugs

#### rocThrust

##### Added
- Updated to match upstream Thrust 1.17.0

#### rocWMMA

##### Added
- Added gemm driver APIs for flow control builtins
- Added benchmark logging systems
- Restructured tests to follow naming convention. Added macros for test generation

##### Changed
- Changed CMake to accomodate the modified test infrastructure
- Fine tuned the multi-block kernels with and without lds
- Adjusted Maximum Vector Width to dWordx4 Width
- Updated Efficiencies to display as whole number percentages
- Updated throughput from GFlops/s to TFlops/s
- Reset the ad-hoc tests to use smaller sizes
- Modified the output validation to use CPU-based implementation against rocWMMA
- Modified the extended vector test to return error codes for memory allocation failures
### Development Tools
No notable changes in this release for development tools, including the compiler, profiler, and debugger.

### Deployment and Management Tools
No notable changes in this release for deployment and management tools.

## Older ROCm™ Releases
For release information for older ROCm™ releases, refer to [CHANGELOG](CHANGELOG.md).

# ROCm™ 5.2.3
This repository contains the manifest file for ROCm™ releases, changelogs and release information. The file default.xml contains information all the repositories and the associated commit use to build the current ROCm release. The default.xml file uses the repo Manifest format.

# Release Notes
The ROCm v5.2.3 is a patch release. The details are listed below. Highlights of this release include a bump in RCCL
version compatibility and minor bug fixes in the HIP Runtime. Additionally, ROCm releases will return to use of the 
[ROCm](https://github.com/RadeonOpenCompute/ROCm) repository for version controlled release notes henceforth. This 
release of ROCm™ is validated with the AMDGPU version 22.20.1.

All users of ROCm™ 5.2.1 and below are encouraged to upgrade. Please visit https://docs.amd.com for all documentation associated with this release. 

## HIP and Other Runtimes

### HIP Runtime

#### Fixes
 - A bug was discovered in the hip graph capture implementation in ROCm™ 5.2.0. If the same kernel is called twice
 (with different argument values) in a graph capture, the implementation was only keeping the argument values for 
 the second kernel call. This bug has now been fixed.
 - A bug was introduced in the hiprtc implementation in ROCm™ 5.2.0. Due to the bug, hiprtcGetLoweredName would fail
 for named expressions that had a whitespace in it. For example, the named expression '''my_sqrt<complex<double>>''' would
 pass but '''my_sqrt<complex<double>>''' would fail. This bug has now been fixed.

### [ROCT-Thunk-Interface](https://github.com/RadeonOpenCompute/ROCT-Thunk-Interface)
No changes in this release.
 
### [ROCR Runtime](https://github.com/RadeonOpenCompute/ROCR-Runtime)
No changes in this release.

### [ROCm-OpenCL-Runtime](https://github.com/RadeonOpenCompute/ROCm-OpenCL-Runtime)
No changes in this release.

### [ROCm Device Libs](https://github.com/RadeonOpenCompute/ROCm-Device-Libs)
No changes in this release.

### [atmi](https://github.com/RadeonOpenCompute/atmi)
No changes in this release.

## ROCm Libraries

### [Tensile](https://github.com/ROCmSoftwarePlatform/Tensile)
No changes in this release.

### [rocBLAS](https://github.com/ROCmSoftwarePlatform/rocBLAS)
No changes in this release.

### [hipBLAS](https://github.com/ROCmSoftwarePlatform/hipBLAS)
No changes in this release.

### [rocSOLVER](https://github.com/ROCmSoftwarePlatform/rocSOLVER)
No changes in this release.

### [hipSOLVER](https://github.com/ROCmSoftwarePlatform/hipSOLVER)
No changes in this release.

### [rocWMMA](https://github.com/ROCmSoftwarePlatform/rocWMMA)
No changes in this release.

### [rocFFT](https://github.com/ROCmSoftwarePlatform/rocFFT)
No changes in this release.

### [hipFFT](https://github.com/ROCmSoftwarePlatform/hipFFT)
No changes in this release.

### [rocPRIM](https://github.com/ROCmSoftwarePlatform/rocPRIM)
No changes in this release.

### [rocCUB](https://github.com/ROCmSoftwarePlatform/hipCUB)
No changes in this release.

### [rocThrust](https://github.com/ROCmSoftwarePlatform/rocThrust)
No changes in this release.

### [rocSPARSE](https://github.com/ROCmSoftwarePlatform/rocSPARSE)
No changes in this release.

### [hipSPARSE](https://github.com/ROCmSoftwarePlatform/hipSPARSE)
No changes in this release.

### [rocALUTION](https://github.com/ROCmSoftwarePlatform/rocALUTION)
No changes in this release.

### [rocRAND](https://github.com/ROCmSoftwarePlatform/rocRAND)
No changes in this release.

### [MIOpenGEMM](https://github.com/ROCmSoftwarePlatform/MIOpenGEMM)
No changes in this release.

### [MIOpen](https://github.com/ROCmSoftwarePlatform/MIOpen)
No changes in this release.

### [MIVisionX](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX)
No changes in this release.

### [AMDMIGraphx](https://github.com/ROCmSoftwarePlatform/AMDMIGraphx)
No changes in this release.

### [hipfort](https://github.com/ROCmSoftwarePlatform/hipfort)
No changes in this release.

### [RCCL](https://github.com/ROCmSoftwarePlatform/rccl)

#### Added
- Compatibility with NCCL 2.12.10
- Packages for test and benchmark executables on all supported OSes using CPack.
- Adding custom signal handler - opt-in with RCCL_ENABLE_SIGNALHANDLER=1
  - Additional details provided if Binary File Descriptor library (BFD) is pre-installed
- Adding experimental support for using multiple ranks per device
  - Requires using a new interface to create communicator (ncclCommInitRankMulti), please
    refer to the interface documentation for details.
  - To avoid potential deadlocks, user might have to set an environment variables increasing
    the number of hardware queues (e.g. export GPU_MAX_HW_QUEUES=16)
- Adding support for reusing ports in NET/IB channels
  - Opt-in with NCCL_IB_SOCK_CLIENT_PORT_REUSE=1 and NCCL_IB_SOCK_SERVER_PORT_REUSE=1
  - When "Call to bind failed : Address already in use" error happens in large-scale AlltoAll
    (e.g., >=64 MI200 nodes), users are suggested to opt-in either one or both of the options
    to resolve the massive port usage issue
  - Avoid using NCCL_IB_SOCK_SERVER_PORT_REUSE when NCCL_NCHANNELS_PER_NET_PEER is tuned >1
#### Removed
- Removed experimental clique-based kernels

## Development Tools

### [LLVM](https://github.com/RadeonOpenCompute/llvm-project)
#### Fixes
- The compiler resolves an issue with usage of the __shfl_sync intrinsic when some of the input are not initialized by the application code base.

### [HIPCC](https://github.com/ROCm-Developer-Tools/HIPCC)
No changes in this release.

### [HIPIFY](https://github.com/ROCm-Developer-Tools/HIPIY)
No changes in this release.

### [ROC Profiler](https://github.com/ROCm-Developer-Tools/rocprofiler)
No changes in this release.

### [ROC Tracer](https://github.com/ROCm-Developer-Tools/roctracer)
No changes in this release.

### [ROCgdb](https://github.com/ROCm-Developer-Tools/ROCgdb)
No changes in this release.

### [ROCdbgapi](https://github.com/ROCm-Developer-Tools/ROCdbgapi)
No changes in this release.

### [Radeon Compute Profiler](https://github.com/GPUOpen-Tools/radeon_compute_profiler)
No changes in this release.

### [AOMP](https://github.com/ROCm-Developer-Tools/aomp)
No changes in this release.

### [AOMP Extras](https://github.com/ROCm-Developer-Tools/aomp-extras)
No changes in this release.

### [flang](https://github.com/ROCm-Developer-Tools/flang)
No changes in this release.

### [ROCm CMake](https://github.com/RadeonOpenCompute/rocm-cmake)
No changes in this release.

### [clang-ocl](https://github.com/RadeonOpenCompute/clang-ocl)
No changes in this release.

### [ROCm CompilerSupport](https://github.com/RadeonOpenCompute/ROCm-CompilerSupport)
No changes in this release.

### [rocr_debug_agent](https://github.com/ROCm-Developer-Tools/rocr_debug_agent)
No changes in this release.

### [half](https://github.com/ROCmSoftwarePlatform/half)
No changes in this release.

## Deployment and Management Tools

### [ROCm Info](https://github.com/RadeonOpenCompute/rocminfo)
No changes in this release.

### [ROCm Data Center Tool](https://github.com/RadeonOpenCompute/rdc)
No changes in this release.

### [ROCmValidationSuite](https://github.com/ROCm-Developer-Tools/ROCmValidationSuite)
No changes in this release.

### [ROCm System Management Interface (ROCm SMI) Library](https://github.com/RadeonOpenCompute/rocm_smi_lib)
No changes in this release.

### [rocm_bandwidth_test](https://github.com/ROCm-Developer-Tools/rocm_bandwidth_test)
No changes in this release.

# Older ROCm™ Releases
For release information for older ROCm™ releases, please visit the [CHANGELOG](CHANGELOG.md).


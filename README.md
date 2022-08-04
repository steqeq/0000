# ROCm™ 5.2.3
This repository contains the manifest file for ROCm™ releases, changelogs and release information. The file default.xml contains information all the repositories and the associated commit use to build the current ROCm release. The default.xml file uses the repo Manifest Format.

# Release Notes
5.2.3 is a patch release for ROCm™. The details are listed below. Highlights of this release include a bump in RCCL version compatibility and minor bug fixes in the HIP Runtime. This release of ROCm™ is validated with the AMDGPU version 22.20.1.

All users of ROCm™ 5.2.1 and below are encouraged to upgrade. Please visit https://docs.amd.com for all documentation associated with this release. 



## HIP and Other Runtimes

### ROCR Runtime
No changes in this release.
### HIP Runtime
#### Fixes
 - A bug was discovered in the hip graph capture implementation in ROCm 5.2.0. If the same kernel is called twice
 (with different argument values) in a graph capture, the implementation was only keeping the argument values for 
 the second kernel call. This bug has now been fixed.
 - A bug was introduced in the hiprtc implementation in ROCm 5.2.0. Due to the bug, hiprtcGetLoweredName would fail
 for named expressions that had a whitespace in it. For example, the named expression "my_sqrt<complex<double>>" would
 pass but "my_sqrt<complex<double> >" would fail. This bug has now been fixed.
### ROCm-OpenCL-Runtime
No changes in this release.

## ROCm Libraries

### [Tensile](https://github.com/ROCmSoftwarePlatform/Tensile)
No changes in this release.

### rocBLAS(https://github.com/ROCmSoftwarePlatform/rocBLAS)
No changes in this release.

### [hipBLAS](https://github.com/ROCmSoftwarePlatform/hipBLAS)
No changes in this release.

### [rocSOLVER](https://github.com/ROCmSoftwarePlatform/rocSOLVER)
No changes in this release.

### [hipSOLVER](https://github.com/ROCmSoftwarePlatform/hipSOLVER)
No changes in this release.

### [rocWMMA](https://github.com/ROCmSoftwarePlatform/rocWMMA)
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

###

### [AOMP]()
No changes in this release.

## Deployment and Management Tools
No changes in this release.

# Older ROCm™ Releases
For release information for older ROCm™ releases, please visit the [CHANGELOG](CHANGELOG.md).


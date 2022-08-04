# ROCm™ 5.2.3
This repository contains the manifest file for ROCm™ releases, changelogs and release information. The file default.xml contains information all the repositories and the associated commit use to build the current ROCm release. The default.xml file uses the repo Manifest Format.

# Release Notes
5.2.3 is a patch release for ROCm™. The details are listed below. Highlights of this release include a bump in RCCL version and minor bug fixes. This release of ROCm™ is validated with the AMDGPU version 22.20.1.

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
 for named expressions that had a whitespace in it. For example the named expression "my_sqrt<complex<double>>" would
 pass but "my_sqrt<complex<double> >" would fail. This bug has now been fixed.
### ROCm-OpenCL-Runtime
No changes in this release.

## HIP Libraries

### [Tensile](https://github.com/ROCmSoftwarePlatform/Tensile)
No changes in this release.
### rocBLAS
No changes in this release.

### hipBLAS
No changes in this release.

### RCCL

## Development Tools
Reference fixed issues.
## Deployment Tools
No changes in this release.

# Older ROCm™ Releases
For release information for older ROCm™ releases, please visit the [CHANGELOG](CHANGELOG.md).


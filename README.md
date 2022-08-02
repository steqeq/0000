
# ROCm™ [ROCmVer]
This repository contains the manifest file for ROCm releases, changelogs and release information. The file default.xml contains information all the repositories and the associated commit use to build the current ROCm release. The default.xml file uses the repo Manifest Format.

# Release Notes
ROCm™ [ROCmVer] is a bug fix point release. The highlights are shown below. All users of ROCm™ 5.2.1 and below are encouraged to upgrade. Please visit https://docs.amd.com for all documentation associated with this release.
## AMDGPU Kernel Drivers
Reference fixed issue
## HIP Runtime
Reference fixed issue
## HIP Libraries
### rocBLAS
#### Fixed
- Improved logic to #include <filesystem> vs <experimental/filesystem>.
- install.sh -s option to build rocblas as a static library.
- dot function now sets the device results asynchronously for N <= 0
### hipBLAS

No changes in this release.
## Development Tools
Reference fixed issues.
## Deployment Tools
No changes in this release.

# Older ROCm™ Releases
For release information for older ROCm™ releases, please visit the [CHANGELOG](CHANGELOG.md).

[ROCmVer] 5.2.2
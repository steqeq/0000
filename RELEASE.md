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

The ROCm 6.0.2 point release consists of minor bug fixes to improve the stability of MI300 GPU applications. This release introduces several new driver features for system qualification on our partner server offerings. 

### Library changes in ROCm 6.0.2

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

#### hipFFT 1.0.13

hipFFT 1.0.13 for ROCm 6.0.2

##### Changes

* Removed the Git submodule for shared files between rocFFT and hipFFT; instead, just copy the files
 over (this should help simplify downstream builds and packaging)

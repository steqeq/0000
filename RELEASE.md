# ROCm 6.1.1 release highlights

<!-- Disable lints since this is an auto-generated file.    -->
<!-- markdownlint-disable blanks-around-headers             -->
<!-- markdownlint-disable no-duplicate-header               -->
<!-- markdownlint-disable no-blanks-blockquote              -->
<!-- markdownlint-disable ul-indent                         -->
<!-- markdownlint-disable no-trailing-spaces                -->

<!-- spellcheck-disable -->

ROCm™ 6.1.1 introduces minor fixes and improvements to some tools and libraries.

In addition, ROCm 6.1.1 has been tested against a pre-release version of Ubuntu 22.04.5 (kernel 6.8).

## AMD SMI

AMD SMI for ROCm 6.1.1

See the [detailed changelog](https://github.com/ROCm/amdsmi/blob/develop/CHANGELOG.md) with code samples for more information.

### Additions

- Added deferred error correctable counts to `amd-smi metric -ecc -ecc-blocks`.

### Changes

- Updated the output of `amd-smi metric --ecc-blocks` to show counters available from blocks.
- Updated the output of `amd-smi metric --clock` to reflect each engine.
- Updated the output of `amd-smi topology --json` to align with output reported by host and guest systems.

### Fixes

- Fixed `amd-smi metric --clock`'s clock lock status and deep sleep status.
- Fixed an issue that would cause an error when attempting to reset non-AMD GPUs.
- Fixed `amd-smi metric --pcie` and `amdsmi_get_pcie_info()` when using Navi 32 and Navi 31 hardware to prevent "UNKNOWN" reports.
- Fixed the output results of `amd-smi process` when getting processes running on a device.

### Removals

- Removed the `amdsmi_get_gpu_process_info` API from the Python library. It was removed from the C library in an earlier release.

### Known issues

- `amd-smi bad-pages` can result in a `ValueError: Null pointer access` error when using certain PMU firmware versions.

## HIPCC

HIPCC for ROCm 6.1.1

### Changes

- **Upcoming:** ROCm 6.2 will enable use of compiled binaries `hipcc.bin` and `hipconfig.bin` by default. No action is needed by users; you may continue calling high-level Perl scripts `hipcc` and `hipconfig`. `hipcc.bin` and `hipconfig.bin` will be invoked by the high-level Perl scripts. To revert to the previous behavior and invoke `hipcc.pl` and `hipconfig.pl`, set the `HIP_USE_PERL_SCRIPTS` environment variable to `1`.
- **Upcoming:** ROCm 6.3 will remove high-level Perl scripts `hipcc` and `hipconfig`. This release will remove the `HIP_USE_PERL_SCRIPTS` environment variable. It will rename `hipcc.bin` and `hipconfig.bin` to `hipcc` and `hipconfig` respectively. No action is needed by the users. To revert to the previous behavior, invoke `hipcc.pl` and `hipconfig.pl` explicitly.
- **Upcoming:** ROCm 7.0 will remove `hipcc.pl` and `hipconfig.pl`.

## HIPIFY

HIPIFY for ROCm 6.1.1

### Additions

* Added support for LLVM 18.1.2.
* Added support for cuDNN 9.0.0.
* Added new options:
  * `--clang-resource-directory` to specify the clang resource path (the path to the parent folder for the `include` folder that contains `__clang_cuda_runtime_wrapper.h` and other header files used during the hipification process).

## hipSOLVER 2.1.1

hipSOLVER 2.1.1 for ROCm 6.1.1

### Changes

- `BUILD_WITH_SPARSE` now defaults to OFF on Windows.

### Fixes

- Fixed benchmark client build when `BUILD_WITH_SPARSE` is OFF.

## rocFFT 1.0.27

rocFFT 1.0.27 for ROCm 6.1.1

### Additions

- Enabled multi-GPU testing on systems without direct GPU-interconnects.

### Fixes

- Fixed kernel launch failure when executing very large odd-length real-complex transforms.

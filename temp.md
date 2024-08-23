## Components

The following table lists ROCm components and their individual versions for ROCm 6.2.0. Find an overview of officially
supported versions of ROCm components, third-party libraries, and frameworks in the
[Compatibility matrix](https://rocm.docs.amd.com/en/latest/release/docs/6.2.0/compatibility/compatibility-matrix).

| Category | Group | Name | Version |   |
|----------|-------|------|---------|:-:|
| **Libraries** | **Machine learning and computer vision** | [Composable Kernel](https://rocm.docs.amd.com/projects/composable_kernel/en/docs/6.2.0) | 1.1.0 | [{fab}`github fa-lg`](https://github.com/ROCm/composable_kernel/releases/tag/rocm-6.2.0) |
|  |  | [MIGraphX](https://rocm.docs.amd.com/projects/AMDMIGraphX/en/docs/6.2.0) | 2.9&nbsp;&Rightarrow;&nbsp;[2.10](migraphx-2-10-0) | [{fab}`github fa-lg`](https://github.com/ROCm/AMDMIGraphX/releases/tag/rocm-6.2.0) |
|  |  | [MIOpen](https://rocm.docs.amd.com/projects/MIOpen/en/docs/6.2.0) | 3.1.0&nbsp;&Rightarrow;&nbsp;[3.2.0](miopen-3-2-0) | [{fab}`github fa-lg`](https://github.com/ROCm/MIOpen/releases/tag/rocm-6.2.0) |
|  |  | [MIVisionX](https://rocm.docs.amd.com/projects/MIVisionX/en/docs/6.2.0) | 2.5.0&nbsp;&Rightarrow;&nbsp;[3.0.0](mivisionx-3-0-0) | [{fab}`github fa-lg`](https://github.com/ROCm/MIVisionX/releases/tag/rocm-6.2.0) |
|  |  | [rocAL](https://rocm.docs.amd.com/projects/rocAL/en/docs/6.2.0) | 2.0.0 | [{fab}`github fa-lg`](https://github.com/ROCm/rocAL/releases/tag/rocm-6.2.0) |
|  |  | [rocDecode](https://rocm.docs.amd.com/projects/rocDecode/en/docs/6.2.0) | 0.6.0 | [{fab}`github fa-lg`](https://github.com/ROCm/rocDecode/releases/tag/rocm-6.2.0) |
|  |  | [rocPyDecode](https://rocm.docs.amd.com/projects/rocPyDecode/en/docs/6.2.0) | 0.1.0 | [{fab}`github fa-lg`](https://github.com/ROCm/rocPyDecode/releases/tag/rocm-6.2.0) |
|  |  | [RPP](https://rocm.docs.amd.com/projects/rpp/en/docs/6.2.0) | 1.5.0&nbsp;&Rightarrow;&nbsp;[1.8.0](rpp-1-8-0) | [{fab}`github fa-lg`](https://github.com/ROCm/rpp/releases/tag/rocm-6.2.0) |
|  | **Communication** | [rccl](https://rocm.docs.amd.com/projects/rccl/en/docs/6.2.0) | 2.18.6&nbsp;&Rightarrow;&nbsp;[2.20.5](rccl-2-20-5) | [{fab}`github fa-lg`](https://github.com/ROCm/rccl/releases/tag/rocm-6.2.0) |
|  | **Math** | [hipBLAS](https://rocm.docs.amd.com/projects/hipBLAS/en/docs/6.2.0) | 2.1.0&nbsp;&Rightarrow;&nbsp;[2.2.0](hipblas-2-2-0) | [{fab}`github fa-lg`](https://github.com/ROCm/hipBLAS/releases/tag/rocm-6.2.0) |
|  |  | [hipBLASLt](https://rocm.docs.amd.com/projects/hipBLASLt/en/docs/6.2.0) | 0.7.0&nbsp;&Rightarrow;&nbsp;[0.8.0](hipblaslt-0-8-0) | [{fab}`github fa-lg`](https://github.com/ROCm/hipBLASLt/releases/tag/rocm-6.2.0) |
|  |  | [hipFFT](https://rocm.docs.amd.com/projects/hipFFT/en/docs/6.2.0) | [1.0.14](hipfft-1-0-14) | [{fab}`github fa-lg`](https://github.com/ROCm/hipFFT/releases/tag/rocm-6.2.0) |
|  |  | [hipfort](https://rocm.docs.amd.com/projects/hipfort/en/docs/6.2.0) | 0.4-0 | [{fab}`github fa-lg`](https://github.com/ROCm/hipfort/releases/tag/rocm-6.2.0) |
|  |  | [hipRAND](https://rocm.docs.amd.com/projects/hipRAND/en/docs/6.2.0) | 2.10.17&nbsp;&Rightarrow;&nbsp;[2.11.0](hiprand-2-11-0) | [{fab}`github fa-lg`](https://github.com/ROCm/hipRAND/releases/tag/rocm-6.2.0) |
|  |  | [hipSOLVER](https://rocm.docs.amd.com/projects/hipSOLVER/en/docs/6.2.0) | 2.1.1&nbsp;&Rightarrow;&nbsp;[2.2.0](hipsolver-2-2-0) | [{fab}`github fa-lg`](https://github.com/ROCm/hipSOLVER/releases/tag/rocm-6.2.0) |
|  |  | [hipSPARSE](https://rocm.docs.amd.com/projects/hipSPARSE/en/docs/6.2.0) | 3.0.1&nbsp;&Rightarrow;&nbsp;[3.1.1](hipsparse-3-1-1) | [{fab}`github fa-lg`](https://github.com/ROCm/hipSPARSE/releases/tag/rocm-6.2.0) |
|  |  | [hipSPARSELt](https://rocm.docs.amd.com/projects/hipSPARSELt/en/docs/6.2.0) | 0.2.0&nbsp;&Rightarrow;&nbsp;[0.2.1](hipsparselt-0-2-1) | [{fab}`github fa-lg`](https://github.com/ROCm/hipSPARSELt/releases/tag/rocm-6.2.0) |
|  |  | [rocALUTION](https://rocm.docs.amd.com/projects/rocALUTION/en/docs/6.2.0) | 3.1.1&nbsp;&Rightarrow;&nbsp;[3.2.0](rocalution-3-2-0) | [{fab}`github fa-lg`](https://github.com/ROCm/rocALUTION/releases/tag/rocm-6.2.0) |
|  |  | [rocBLAS](https://rocm.docs.amd.com/projects/rocBLAS/en/docs/6.2.0) | 4.1.0&nbsp;&Rightarrow;&nbsp;[4.2.0](rocblas-4-2-0) | [{fab}`github fa-lg`](https://github.com/ROCm/rocBLAS/releases/tag/rocm-6.2.0) |
|  |  | [rocFFT](https://rocm.docs.amd.com/projects/rocFFT/en/docs/6.2.0) | 1.0.27&nbsp;&Rightarrow;&nbsp;[1.0.28](rocfft-1-0-28) | [{fab}`github fa-lg`](https://github.com/ROCm/rocFFT/releases/tag/rocm-6.2.0) |
|  |  | [rocRAND](https://rocm.docs.amd.com/projects/rocRAND/en/docs/6.2.0) | 3.0.0&nbsp;&Rightarrow;&nbsp;[3.1.0](rocrand-3-1-0) | [{fab}`github fa-lg`](https://github.com/ROCm/rocRAND/releases/tag/rocm-6.2.0) |
|  |  | [rocSOLVER](https://rocm.docs.amd.com/projects/rocSOLVER/en/docs/6.2.0) | 3.25.0&nbsp;&Rightarrow;&nbsp;[3.26.0](rocsolver-3-26-0) | [{fab}`github fa-lg`](https://github.com/ROCm/rocSOLVER/releases/tag/rocm-6.2.0) |
|  |  | [rocSPARSE](https://rocm.docs.amd.com/projects/rocSPARSE/en/docs/6.2.0) | 3.1.1&nbsp;&Rightarrow;&nbsp;[3.2.0](rocsparse-3-2-0) | [{fab}`github fa-lg`](https://github.com/ROCm/rocSPARSE/releases/tag/rocm-6.2.0) |
|  |  | [rocWMMA](https://rocm.docs.amd.com/projects/rocWMMA/en/docs/6.2.0) | 1.4.0&nbsp;&Rightarrow;&nbsp;[1.5.0](rocwmma-1-5-0) | [{fab}`github fa-lg`](https://github.com/ROCm/rocWMMA/releases/tag/rocm-6.2.0) |
|  |  | [Tensile](https://rocm.docs.amd.com/projects/tensile/en/docs/6.2.0) | 4.40.0&nbsp;&Rightarrow;&nbsp;[4.41.0](tensile-4-41-0) | [{fab}`github fa-lg`](https://github.com/ROCm/tensile/releases/tag/rocm-6.2.0) |
|  | **Primitives** | [hipCUB](https://rocm.docs.amd.com/projects/hipCUB/en/docs/6.2.0) | 3.1.0&nbsp;&Rightarrow;&nbsp;[3.2.0](hipcub-3-2-0) | [{fab}`github fa-lg`](https://github.com/ROCm/hipCUB/releases/tag/rocm-6.2.0) |
|  |  | [hipTensor](https://rocm.docs.amd.com/projects/hipTensor/en/docs/6.2.0) | 1.2.0&nbsp;&Rightarrow;&nbsp;[1.3.0](hiptensor-1-3-0) | [{fab}`github fa-lg`](https://github.com/ROCm/hipTensor/releases/tag/rocm-6.2.0) |
|  |  | [rocPRIM](https://rocm.docs.amd.com/projects/rocPRIM/en/docs/6.2.0) | 3.1.0&nbsp;&Rightarrow;&nbsp;[3.2.0](rocprim-3-2-0) | [{fab}`github fa-lg`](https://github.com/ROCm/rocPRIM/releases/tag/rocm-6.2.0) |
|  |  | [rocThrust](https://rocm.docs.amd.com/projects/rocThrust/en/docs/6.2.0) | 3.0.0&nbsp;&Rightarrow;&nbsp;[3.1.0](rocthrust-3-1-0) | [{fab}`github fa-lg`](https://github.com/ROCm/rocThrust/releases/tag/rocm-6.2.0) |
| **Tools** | **Development** | [HIPIFY](https://rocm.docs.amd.com/projects/HIPIFY/docs/6.2.0) | 17.0.0&nbsp;&Rightarrow;&nbsp;[18.0.0](hipify-18-0-0) | [{fab}`github fa-lg`](https://github.com/ROCm/HIPIFY/releases/tag/rocm-6.2.0) |
|  |  | [ROCdbgapi](https://rocm.docs.amd.com/projects/ROCdbgapi/en/docs/6.2.0) | 0.71.0&nbsp;&Rightarrow;&nbsp;[0.76.0](rocdbgapi-0-76-0) | [{fab}`github fa-lg`](https://github.com/ROCm/ROCdbgapi/releases/tag/rocm-6.2.0) |
|  |  | [ROCm CMake](https://rocm.docs.amd.com/projects/rocm-cmake/en/docs/6.2.0) | 0.12.0&nbsp;&Rightarrow;&nbsp;[0.13.0](rocm-cmake-0-13-0) | [{fab}`github fa-lg`](https://github.com/ROCm/rocm-cmake/releases/tag/rocm-6.2.0) |
|  |  | [ROCm Debugger (ROCgdb)](https://rocm.docs.amd.com/projects/rocm-cmake/en/docs/6.2.0) | 13&nbsp;&Rightarrow;&nbsp;[15](rocgdb-15) | [{fab}`github fa-lg`](https://github.com/ROCm/ROCgdb/releases/tag/rocm-6.2.0) |
|  |  | [ROCr Debug Agent](https://rocm.docs.amd.com/projects/rocr_debug_agent/en/docs/6.2.0) | 2.0.3 | [{fab}`github fa-lg`](https://github.com/ROCm/rocr_debug_agent/releases/tag/rocm-6.2.0) |
|  | **Performance** | [Omniperf](https://rocm.docs.amd.com/projects/omniperf/en/docs/6.2.0) | 2.0.1 | [{fab}`github fa-lg`](https://github.com/ROCm/omniperf/releases/tag/rocm-6.2.0) |
|  |  | [Omnitrace](https://rocm.docs.amd.com/projects/omnitrace/en/docs/6.2.0) | 1.11.2 | [{fab}`github fa-lg`](https://github.com/ROCm/omnitrace/releases/tag/rocm-6.2.0) |
|  |  | [ROCm Bandwidth Test](https://rocm.docs.amd.com/projects/rocm_bandwidth_test/en/docs/6.2.0) | 1.4.0 | [{fab}`github fa-lg`](https://github.com/ROCm/rocm_bandwidth_test/releases/tag/rocm-6.2.0) |
|  |  | [ROCProfiler](https://rocm.docs.amd.com/projects/ROCProfiler/en/docs/6.2.0) | 2.0.0&nbsp;&Rightarrow;&nbsp;[2.0.0](rocprofiler-2-0-0) | [{fab}`github fa-lg`](https://github.com/ROCm/rocm_bandwidth_test/releases/tag/rocm-6.2.0) |
|  |  | [ROCProfiler-SDK](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs/6.2.0) | 0.4.0 | [{fab}`github fa-lg`](https://github.com/ROCm/rocm_bandwidth_test/releases/tag/rocm-6.2.0) |
|  |  | [ROCTracer](https://rocm.docs.amd.com/projects/ROCTracer/en/docs/6.2.0) | 4.1.0 | [{fab}`github fa-lg`](https://github.com/ROCm/rocm_bandwidth_test/releases/tag/rocm-6.2.0) |
|  | **System** | [AMD SMI](https://rocm.docs.amd.com/projects/amdsmi/en/docs/6.2.0) | 24.5.2&nbsp;&Rightarrow;&nbsp;[24.6.1](amd-smi-24-6-1) | [{fab}`github fa-lg`](https://github.com/ROCm/rdc/releases/tag/rocm-6.2.0) |
|  |  | [rocminfo](https://rocm.docs.amd.com/projects/rdc/en/docs/6.2.0) | 1.0.0 | [{fab}`github fa-lg`](https://github.com/ROCm/rdc/releases/tag/rocm-6.2.0) |
|  |  | [ROCm Data Center Tool](https://rocm.docs.amd.com/projects/rdc/en/docs/6.2.0) | 0.3.0&nbsp;&Rightarrow;&nbsp;[1.0.0](rocm-data-center-tool-1-0-0) | [{fab}`github fa-lg`](https://github.com/ROCm/rdc/releases/tag/rocm-6.2.0) |
|  |  | [ROCm SMI](https://rocm.docs.amd.com/projects/rdc/en/docs/6.2.0) | 7.2.0 | [{fab}`github fa-lg`](https://github.com/ROCm/rdc/releases/tag/rocm-6.2.0) |
|  |  | [ROCm Validation Suite](https://rocm.docs.amd.com/projects/rdc/en/docs/6.2.0) | 1.0 | [{fab}`github fa-lg`](https://github.com/ROCm/rdc/releases/tag/rocm-6.2.0) |
|  |  | [TransferBench](https://rocm.docs.amd.com/projects/rdc/en/docs/6.2.0) | 1.5.0 | [{fab}`github fa-lg`](https://github.com/ROCm/rdc/releases/tag/rocm-6.2.0) |
|  | **Compilers** | [hipCC](https://rocm.docs.amd.com/projects/hipCC/en/docs/6.2.0) | 1.0.0&nbsp;&Rightarrow;&nbsp;[1.1.1](hipcc-1-1-1) | [{fab}`github fa-lg`](https://github.com/ROCm/llvm-project/releases/tag/rocm-6.2.0) |
|  |  | [llvm-project](https://rocm.docs.amd.com/projects/llvm-project/en/docs/6.2.0) | 17.0.0&nbsp;&Rightarrow;&nbsp;[18.0.0](llvm-project-18-0-0) | [{fab}`github fa-lg`](https://github.com/ROCm/llvm-project/releases/tag/rocm-6.2.0) |
| **Runtimes** |  | [HIP](https://rocm.docs.amd.com/projects/HIP/en/docs/6.2.0) | 6.1&nbsp;&Rightarrow;&nbsp;[6.2](hip-6-2-0) | [{fab}`github fa-lg`](https://github.com/ROCm/HIP/releases/tag/rocm-6.2.0) |
|  |  | [ROCr Runtime](https://rocm.docs.amd.com/projects/ROCr-Runtime/en/docs/6.2.0) | 6.1&nbsp;&Rightarrow;&nbsp;[6.2](hip-6-2-0) | [{fab}`github fa-lg`](https://github.com/ROCm/ROCR-Runtime/releases/tag/rocm-6.2.0) |

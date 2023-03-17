# HIP
HIP is both AMD's GPU programming langauge extension and the GPU runtime. This page introduces the HIP runtime and other HIP libraries and tools.
## HIP Runtime and Libraries
:::::{grid} 1 1 2 2
:gutter: 1

:::{grid-item-card} HIP Runtime
The HIP Runtime is used to enable GPU acceleration for all HIP language based products.

- [API Reference Manual](https://rocmdocs.amd.com/projects/hipBLAS/en/rtd/)
- [Examples](https://github.com/amd/rocm-examples/tree/develop/HIP-Basic)

:::



:::{grid-item-card} [Math Libraries](./gpu_libraries/math)
HIP Math Libraries support the following domains:

- [API Reference Manual](./gpu_libraries/blas)
- [Changelog](https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/develop/CHANGELOG.md)
- [Examples](https://github.com/amd/rocm-examples/tree/develop/Libraries/rocBLAS)

:::

:::{grid-item-card} [C++ Primitives](./gpu_libraries/c++_primitives)
ROCm template libraries for C++ primitives and algorithms are as follows:

- [API Reference Manual](https://rocprim.readthedocs.io/en/latest/)
- [API Reference Manual](https://rocthrust.readthedocs.io/en/latest/)
- [API Reference Manual](https://hipcub.readthedocs.io/en/latest/)

:::

:::{grid-item-card} Communication Libraries
AMD's C++ library for accelerating mixed-precision matrix multiply-accumulate (MMA)
operations leveraging AMD GPU hardware.

- [API Reference Manual](https://docs.amd.com/bundle/rocWMMA-release-rocm-rel-5.2/page/API_Reference_Guide.html)
- [Changelog](https://github.com/ROCmSoftwarePlatform/rocWMMA/blob/develop/CHANGELOG.md)
- [Examples](https://github.com/ROCmSoftwarePlatform/rocWMMA/tree/develop/samples)

:::

:::{grid-item-card} MIOpen
AMD's C++ library for accelerating mixed-precision matrix multiply-accumulate (MMA)
operations leveraging AMD GPU hardware.

- [API Reference Manual](https://docs.amd.com/bundle/rocWMMA-release-rocm-rel-5.2/page/API_Reference_Guide.html)
- [Changelog](https://github.com/ROCmSoftwarePlatform/rocWMMA/blob/develop/CHANGELOG.md)
- [Examples](https://github.com/ROCmSoftwarePlatform/rocWMMA/tree/develop/samples)

:::

:::{grid-item-card} MIGraphX
AMD's C++ library for accelerating mixed-precision matrix multiply-accumulate (MMA)
operations leveraging AMD GPU hardware.

- [API Reference Manual](https://docs.amd.com/bundle/rocWMMA-release-rocm-rel-5.2/page/API_Reference_Guide.html)
- [Changelog](https://github.com/ROCmSoftwarePlatform/rocWMMA/blob/develop/CHANGELOG.md)
- [Examples](https://github.com/ROCmSoftwarePlatform/rocWMMA/tree/develop/samples)

:::


:::::

## Porting tools

:::::{grid} 1 1 1 1
:gutter: 1

:::{grid-item-card} HIPify
HIPify assists with porting applications from based on CUDA to the HIP Runtime. Supported
CUDA APIs are documented here as well.

- [Reference Manual](https://rocmdocs.amd.com/projects/rocBLAS/en/rtd/)

:::

:::::
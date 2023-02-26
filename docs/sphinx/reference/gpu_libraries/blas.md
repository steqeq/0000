# BLAS

ROCm libraries for BLAS are as follows:

:::::{grid} 1 1 2 2
:gutter: 1

:::{grid-item-card} hipBLAS
hipBLAS is a compatiblity layer for GPU accelerated BLAS optimized for AMD GPUs
via rocBLAS and rocSOLVER. hipBLAS allows for a common interface for other GPU
BLAS libraries. Users adopting ROCm are encouraged to choose the hipBLAS
interface over rocBLAS.

- [Changelog](https://github.com/ROCmSoftwarePlatform/hipBLAS/blob/develop/CHANGELOG.md)
- User Guide
- API Guide
- Examples

:::

:::{grid-item-card} rocBLAS
rocBLAS is an AMD GPU optimized library for BLAS.

- [Changelog](https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/develop/CHANGELOG.md)
- User Guide
- API Guide
- [Examples](https://github.com/amd/rocm-examples/tree/develop/Libraries/rocBLAS)

:::

:::::

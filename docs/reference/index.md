# Reference material

## ROCm Software Groups

:::::{grid} 1 1 2 2
:gutter: 1

:::{grid-item-card}
**[HIP](./hip.md)**

HIP is both AMD's GPU programming language extension and the GPU runtime.

* {doc}`HIP <hip:index>`
* [HIP Examples](https://github.com/amd/rocm-examples/tree/develop/HIP-Basic)
* {doc}`HIPIFY <hipify:index>`

:::

:::{grid-item-card}
**[Math Libraries](./libraries/gpu_libraries/math.md)**

HIP Math Libraries support the following domains:

* [Linear Algebra Libraries](./libraries/gpu_libraries/math_linear_algebra.md)
* [Fast Fourier Transforms](./libraries/gpu_libraries/math_fft.md)
* [Random Numbers](./libraries/gpu_libraries/rand.md)

:::

:::{grid-item-card}
**[C++ Primitive Libraries](./libraries/gpu_libraries/c++_primitives.md)**

ROCm template libraries for C++ primitives and algorithms are as follows:

* {doc}`rocPRIM <rocprim:index>`
* {doc}`rocThrust <rocthrust:index>`
* {doc}`hipCUB <hipcub:index>`
* {doc}`hipTensor <hiptensor:index>`

:::

:::{grid-item-card} [Communication Libraries](./libraries/gpu_libraries/communication.md)
Inter and intra-node communication is supported by the following projects:

* {doc}`RCCL <rccl:index>`

:::

:::{grid-item-card}
**[Artificial intelligence](../rocm_ai.md)**

Libraries related to AI.

* {doc}`MIOpen <miopen:index>`
* {doc}`Composable Kernel <composable_kernel:index>`
* {doc}`MIGraphX <amdmigraphx:index>`
* {doc}`MIVisionX <mivisionx:README>`
* {doc}`rocAL <rocal:README>`
:::

:::{grid-item-card}
**[OpenMP](./openmp/openmp.md)**

* [OpenMP Support Guide](./openmp/openmp.md)

:::

:::{grid-item-card}
**[Compilers and Tools](./compilers_tools/index.md)**

* [ROCmCC](./rocmcc/rocmcc.md)
* {doc}`ROCdbgapi <rocdbgapi:index>`
* {doc}`ROCgdb <rocgdb:index>`
* {doc}`ROCProfiler <rocprofiler:rocprof>`
* {doc}`ROCTracer <roctracer:index>`

:::

:::{grid-item-card}
**[Management Tools](./compilers_tools/management_tools.md)**

* {doc}`AMD SMI <amdsmi:index>`
* {doc}`ROCm SMI <rocm_smi_lib:index>`
* {doc}`ROCm Data Center Tool <rdc:index>`

:::

:::{grid-item-card}
**[Validation Tools](./compilers_tools/validation_tools.md)**

* {doc}`ROCm Validation Suite <rocmvalidationsuite:index>`
* {doc}`TransferBench <transferbench:index>`

:::

:::{grid-item-card} **GPU Architectures**

* [AMD Instinct MI200](../conceptual/gpu_arch/mi250.md)
* [AMD Instinct MI100](../conceptual/gpu_arch/mi100.md)

:::

:::::

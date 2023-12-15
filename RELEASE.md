# Release notes for AMD ROCm™ 6.0

ROCm 6.0 is a major release with new performance optimizations, expanded frameworks and library
support, and improved developer experience. Initial enablement of all AMD Instinct™ MI300-related
features are delivered in ROCm 6.0. Future releases will provide additional enablement and
optimizations for this new platform.

* FP8 support in PyTorch along with hipBLASLt enables optimized performance.
* Upstream support is now available for popular AI frameworks like TensorFlow, JAX, and PyTorch
* ROCm 6.0 adds support for AI libraries, such as DeepSpeed, ONNX-RT, Jax, and CuPy. HPC
  frameworks are optimized for ROCm, including multi-GPU and multi-server capabilities.
* hipSPARSELt provides AMD's sparse matrix core technique on AMD RDNA/CDNA GPUs, which has
  the potential to speed up AI workloads and inference engines.
* Prepackaged HPC and AI containers are available on [AMD Infinity Hub](https://www.amd.com/en/developer/resources/infinity-hub.html). We have improved
  documentation and tutorials on the [AMD ROCm Docs](https://rocm.docs.amd.com) site.
* AMD has consolidated developer resources and training on our new [AMD ROCm Developer Hub](https://www.amd.com/en/developer/resources/rocm-hub.html).
  ROCm 6.0 has focused on performance tuning in key areas, such as lower precision math and
  attention layers.

For additional details on these release notes, you can refer to the
[Changelog](https://rocm.docs.amd.com/en/develop/about/CHANGELOG.html).

#### OS and GPU support changes

AMD Instinct™ MI300A and MI300X Accelerator support has been enabled for supported operating
systems, excluding RHEL 9.

We've added support for the following operating systems:

* RHEL 9.3
* RHEL 8.9

Note that, of ROCm 6.2, we've planned for end-of-support for the following operating systems:

* Ubuntu 20.04.5
* SLES 15 SP4
* RHEL/CentOS 7.9

#### New ROCm meta package

A new ROCm meta package has been added for easy installation of all ROCm core packages, tools, and libraries.
For example, the following command will install the full ROCm package: `apt-get install rocm` (Ubuntu), or `yum install rocm` (RHEL).

#### Filesystem Hierarchy Standard

ROCm 6.0 fully adopts the Filesystem Hierarchy Standard (FHS) reorganization goals. We've removed
the backward compatibility support for old file locations.

#### Compiler location change

* The installation path of LLVM has been changed from `/opt/rocm-<rel>/llvm` to
  `/opt/rocm-<rel>/lib/llvm`. For backward compatibility, a symbolic link is provided to the old
  location and will be removed in ROCm 7.0 or later.
* The installation path of the device library bitcode has changed from `/opt/rocm-<rel>/amdgcn` to
  `/opt/rocm-<rel>/lib/llvm/lib/clang/<ver>/lib/amdgcn`. For backward compatibility, a symbolic link
  is provided and will be removed in a future release.

#### Documentation

CMake support has been added for documentation in the
[ROCm repository](https://github.com/RadeonOpenCompute/ROCm).

#### AMD Instinct™ MI50 end-of-support notice

AMD Instinct MI50, Radeon Pro VII, and Radeon VII products (collectively gfx906 GPUs) enters
maintenance mode in ROCm 6.0.0.

As outlined in [5.6.0](https://rocm.docs.amd.com/en/docs-5.6.0/release.html), ROCm 5.7 is the
final release for gfx906 GPUs in a fully supported state.

  * Henceforth, no new features and performance optimizations will be supported for the gfx906 GPUs.

  * Bug fixes and critical security patches will continue to be supported for the gfx906 GPUs until Q2
    2024 (end of maintenance \[EOM] will be aligned with the closest ROCm release).

  * Bug fixes will be made up to the next ROCm point release.

  * Bug fixes will not be backported to older ROCm releases for gfx906.

  * Distribution and operating system updates will continue per the ROCm release cadence for gfx906
    GPUs until EOM.

### AMD SMI

* **Integrated the E-SMI library**.
    You can now query CPU-related information directly through AMD SMI. Metrics include power,
    energy, performance, and other system details.

* **Added support for gfx940/941/942 metrics**.
    You can now query MI300 device metrics to get real-time information. Metrics include power,
    temperature, energy, and performance.

* **Added support for compute and memory partitions**.
    You can now view, set and reset partitions on supported platforms. The topology display provides a
    more in-depth look at a device's current configuration.

### HIP

* **New features to improve resource interoperability**.
  * You can now use new fields and structs for external resource interoperability for memory handles
    and buffers, and semaphores.
  * You can use the new environment variable, `HIP_LAUNCH_BLOCKING`, to control serialization on
    kernel execution.
  * Use additional members to HIP struct `hipDeviceProp_t` for surfaces, textures, and device identifiers.
    * LUID (Locally Unique Identifier) is supported for interoperability between devices. In HIP, more
     members are added in the struct `hipDeviceProp_t`, as properties to identify each device.

* **Changes impacting backward compatibility**.
  * We've changed data types for members in `HIP_MEMCPY3D` structure from `unsigned int` to `size_t`.
  * The value of the flag `hipIpcMemLazyEnablePeerAccess` is changed to `0x01`, which was previously
    defined as `0`.
  * Some device property attributes are not currently supported in HIP runtime. In order to maintain
    consistency, the following related enumeration names are changed in `hipDeviceAttribute_t`
    * `hipDeviceAttributeName` is changed to `hipDeviceAttributeUnused1`
    * `hipDeviceAttributeUuid` is changed to `hipDeviceAttributeUnused2`
    * `hipDeviceAttributeArch` is changed to `hipDeviceAttributeUnused3`
    * `hipDeviceAttributeGcnArch` is changed to `hipDeviceAttributeUnused4`
    * `hipDeviceAttributeGcnArchName` is changed to `hipDeviceAttributeUnused5`
  * HIP struct `hipArray` is removed from driver type header to comply with CUDA
  * `hipArray_t` replaces `hipArray*`, as the pointer to array.
    * This allows `hipMemcpyAtoH` and `hipMemcpyHtoA` to have the correct array type which is
      equivalent to corresponding CUDA driver APIs.
  * For additional deprecation information, refer to the changelog.

### hipCUB

* **Additional CUB API support**.
    The hipCUB backend is updated to CUB and Thrust 2.1.

### HIPIFY

* **Enhanced CUDA2HIP document generation**.
    API versions are now listed in the CUDA2HIP documentation. To see if the application binary
    interface (ABI) has changed, refer to the
    [*C* column](https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Runtime_API_functions_supported_by_HIP.html)
    in our API documentation.

* **Hipified rocSPARSE**.
    We've implemented support for the direct hipification of additional cuSPARSE APIs into rocSPARSE
    APIs under the `--roc` option. This covers a major milestone in the roadmap towards complete
    cuSPARSE-to-rocSPARSE hipification.

### hipRAND

* **Official release**.
    hipRAND is now a *standalone project*--it's no longer available as a submodule for rocRAND.

### hipTensor

* **Added architecture support**.
    We've added contraction support for gfx940, gfx941, and gfx942 architectures, and f32 and f64 data
    types.

* **Upgraded testing infrastructure**.
    hipTensor will now support dynamic parameter configuration with input YAML config.

### MIGraphX

* **Added TorchMIGraphX**.
    We introduced a Dynamo backend for Torch, which allows PyTorch to use MIGraphX directly
    without first requiring a model to be converted to the ONNX model format. With a single line of
    code, PyTorch users can utilize the performance and quantization benefits provided by MIGraphX.
    To learn more, see https://github.com/ROCmSoftwarePlatform/torch_migraphx.

* **Boosted overall performance with rocMLIR**.
    We've integrated the rocMLIR library for ROCm-supported RDNA and CDNA GPUs. This
    technology provides MLIR-based convolution and GEMM kernel generation. MIGraphX inclusion of
    rocMLIR in ROCm 6.0 focused on convolutional fusions.

* **Added INT8 support across the MIGraphX portfolio**.
    We now support the INT8 data type. MIGraphX can perform the quantization or ingest
    prequantized models. INT8 support extends to the MIGraphX execution provider for ONNX Runtime.

### ROCgdb

* **Added support for additional GPU architectures**.
  * Navi 3 Series: gfx1100, gfx1101, and gfx1102.
  * MI300 Series: gfx940, gfx941, and gfx942.

### rocm-smi-lib

* **Improved accessibility to GPU partition nodes**.
    You can now view, set, and reset the compute and memory partitions. You'll also get notifications of
    a GPU busy state, which helps you avoid partition set or reset failure.

* **Upgraded GPU metrics version 1.4**.
    The upgraded GPU metrics binary has an improved metric version format with a content version
    appended to it. You can read each metric within the binary without the full `rsmi_gpu_metric_t` data
    structure.

* **Updated GPU index sorting**.
    We made GPU index sorting consistent with other ROCm software tools by optimizing it to use
    `Bus:Device.Function` (BDF) instead of the card number.

### ROCmValidationSuite

* **Added GPU and operating system support**.
    ROCmValidationSuite (RVS) now supports the CBL-Mariner operating system, and Navi 31 and
    Navi 32 GPUs. We've also added GPU Stress Test (GST) support for MI300X GPUs.

### rocProfiler

* **Added option to specify desired RocProf version**.
    You can now use rocProfV1 or rocProfV2 by specifying your desired version, as the legacy rocProf
    (`rocprofv1`) provides the option to use the latest version (`rocprofv2`).

* **Automated the ISA dumping process by Advance Thread Tracer**.
    Advance Thread Tracer (ATT) no longer depends on user-supplied Instruction Set Architecture (ISA)
    and compilation process (using ``hipcc --save-temps``) to dump ISA from the running kernels.

* **Added ATT support for parallel kernels**.
    The automatic ISA dumping process also helps ATT successfully parse multiple kernels running in
    parallel, and provide cycle-accurate occupancy information for multiple kernels at the same time.

### ROCr

* **Support for SDMA link aggregation**.
    If multiple XGMI links are available when making SDMA copies between GPUs, the copy is
    distributed over multiple links to increase peak bandwidth.

### rocThrust

* **Added Thrust 2.1 API support**.
    rocThrust backend is updated to Thrust and CUB 2.1.

### rocWMMA

* **Added new architecture support**.
    We added support for gfx940, gfx941, and gfx942 architectures.

* **Added data type support**.
    We added support for f8, bf8, xf32 data types on supporting architectures, and for bf16 in the HIP RTC
    environment.

* **Added support for the PyTorch kernel plugin**.
    We added awareness of `__HIP_NO_HALF_CONVERSIONS__` to support PyTorch users.

### TransferBench

* **Improved ordering control**.
    You can now set the thread block size (`BLOCK_SIZE`) and the thread block order (`BLOCK_ORDER`)
    in which thread blocks from different transfers are run when using a single stream.

* **Added comprehensive reports**.
    We modified individual transfers to report X Compute Clusters (XCC) ID when `SHOW_ITERATIONS`
    is set to 1.

* **Improved accuracy in result validation**.
    You can now validate results for each iteration instead of just once for all iterations.

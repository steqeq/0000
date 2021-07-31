# AMD ROCm™ v4.3 Release Notes 

This document describes the features, fixed issues, and information about downloading and installing the AMD ROCm™ software. It also covers known issues and deprecations in this release.

- [Supported Operating Systems and Documentation Updates](#Supported-Operating-Systems-and-Documentation-Updates)
  * [Supported Operating Systems](#Supported-Operating-Systems)
  * [ROCm Installation Updates](#ROCm-Installation-Updates)
  * [AMD ROCm Documentation Updates](#AMD-ROCm-Documentation-Updates)

   
- [What\'s New in This Release](#Whats-New-in-This-Release)
  * [HIP Enhancements](#HIP-Enhancements)
  * [ROCm Data Center Tool](#ROCm-Data-Center-Tool)
  * [ROCm Math and Communication Libraries](#ROCm-Math-and-Communication-Libraries)   


- [Fixed Defects](#Fixed-Defects)  

- [Known Issues](#Known-Issues)

- [Deploying ROCm](#Deploying-ROCm)
 
- [Hardware and Software Support](#Hardware-and-Software-Support)

- [Machine Learning and High Performance Computing Software Stack for AMD GPU](#Machine-Learning-and-High-Performance-Computing-Software-Stack-for-AMD-GPU)
  * [ROCm Binary Package Structure](#ROCm-Binary-Package-Structure)
  * [ROCm Platform Packages](#ROCm-Platform-Packages)
  



## ROCm Installation Updates 

### Supported Operating Systems

The AMD ROCm platform is designed to support the following operating systems:

![Screenshot](https://github.com/Rmalavally/ROCm/blob/master/images/OSKernel.PNG)


### Complete Installation of AMD ROCM V4.3 Recommended

Complete uninstallation of previous ROCm versions is required before installing a new version of ROCm. **An upgrade from previous releases to AMD ROCm v4.3 is not supported**. For more information, refer to the AMD ROCm Installation Guide at

https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html

**Note**: AMD ROCm release v3.3 or prior releases are not fully compatible with AMD ROCm v3.5 and higher versions. You must perform a fresh ROCm installation if you want to upgrade from AMD ROCm v3.3 or older to 3.5 or higher versions and vice-versa.

**Note**: *render* group is required only for Ubuntu v20.04. For all other ROCm supported operating systems, continue to use video group. 

* For ROCm v3.5 and releases thereafter, the clinfo path is changed to /opt/rocm/opencl/bin/clinfo. 

* For ROCm v3.3 and older releases, the clinfo path remains /opt/rocm/opencl/bin/x86_64/clinfo. 
 
## ROCm Multi-Version Installation Update

With the AMD ROCm v4.3 release, the following ROCm multi-version installation changes apply:

The meta packages rocm-dkms<version> are now deprecated for multi-version ROCm installs.  For example, rocm-dkms3.7.0, rocm-dkms3.8.0.
 
* Multi-version installation of ROCm should be performed by installing rocm-dev<version> using each of the desired ROCm versions. For example, rocm-dev3.7.0, rocm-dev3.8.0, rocm-dev3.9.0.   

* The rock-dkms loadable kernel modules should be installed using a single rock-dkms package. 

* ROCm v3.9 and above will not set any ldconfig entries for ROCm libraries for multi-version installation.  Users must set LD_LIBRARY_PATH to load the ROCm library version of choice.

**NOTE**: The single version installation of the ROCm stack remains the same. The rocm-dkms package can be used for single version installs and is not deprecated at this time.
	

## Support for Enviornment Modules
	
Environment modules are now supported. This enhancement in the ROCm v4.3 release enables users to switch between ROCm v4.2 and ROCm v4.3 easily and efficiently. 
	
For more information about installing environment modules, refer to
	
https://modules.readthedocs.io/en/latest/
 


# AMD ROCm Documentation Updates

## AMD ROCm Installation Guide 

The AMD ROCm Installation Guide in this release includes:

* Supported Environments

* Installation Instructions 

* HIP Installation Instructions 

For more information, refer to the ROCm documentation website at:

https://rocmdocs.amd.com/en/latest/


## AMD ROCm - HIP Documentation Updates

* HIP Programming Guide v4.3 

* HIP API Guide v4.3

* HIP-Supported CUDA API Reference Guide v4.3
	
* AMD ROCm Compiler Reference Guide v4.3 - *NEW*

* HIP FAQ  

  For more information, refer to

https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-FAQ.html#hip-faq


## ROCm Data Center User and API Guide

* ROCm Data Center Tool User Guide

  - Prometheus (Grafana) Integration with Automatic Node Detection 
	


* ROCm Data Center Tool API Guide


## ROCm SMI API Documentation Updates 

* ROCm SMI API Guide 

 
  
## ROC Debugger User and API Guide 

* ROC Debugger User Guide  

* Debugger API Guide 


## General AMD ROCm Documentation Links

Access the following links for more information:

* For AMD ROCm documentation, see 

  https://rocmdocs.amd.com/en/latest/

* For installation instructions on supped platforms, see

  https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html

* For AMD ROCm binary structure, see

  https://rocmdocs.amd.com/en/latest/Installation_Guide/Software-Stack-for-AMD-GPU.html
  

* For AMD ROCm Release History, see

 https://rocmdocs.amd.com/en/latest/Current_Release_Notes/ROCm-Version-History.html



# What\'s New in This Release

## HIP Enhancements

### HIP Versioning Update

The HIP version definition is updated from the ROCm v4.2 release as follows: 

```
	HIP_VERSION=HIP_VERSION_MAJOR * 10000000 + HIP_VERSION_MINOR * 100000 + 
	HIP_VERSION_PATCH)
```

The HIP version can be queried from a HIP API call


```	
	hipRuntimeGetVersion(&runtimeVersion);	
```
	
**Note**: The version returned will be greater than the version in previous ROCm releases.

For example,
	
### Support for Managed Memory Allocation

HIP now supports and automatically manages Heterogeneous Memory Management (HMM) allocation. The HIP application performs a capability check before making the managed memory API call hipMallocManaged.

**Note**: The _managed_ keyword is unsupported currently. 

```
	int managed_memory = 0;
	HIPCHECK(hipDeviceGetAttribute(&managed_memory,
 	 hipDeviceAttributeManagedMemory,p_gpuDevice));
	if (!managed_memory ) {
  	printf ("info: managed memory access not supported on the device %d\n Skipped\n", p_gpuDevice);
	}
	else {
 	 HIPCHECK(hipSetDevice(p_gpuDevice));
  	HIPCHECK(hipMallocManaged(&Hmm, N * sizeof(T)));
	. . .
	}
```

### Kernel Enqueue Serialization

Developers can control kernel command serialization from the host using the following environment variable,
AMD_SERIALIZE_KERNEL
	
* AMD_SERIALIZE_KERNEL = 1, Wait for completion before enqueue,

* AMD_SERIALIZE_KERNEL = 2, Wait for completion after enqueue,

* AMD_SERIALIZE_KERNEL = 3, Both.

This environment variable setting enables HIP runtime to wait for GPU idle before/after any GPU command.


### NUMA-aware Host Memory Allocation
	
The Non-Uniform Memory Architecture (NUMA) policy determines how memory is allocated and selects a CPU closest to each GPU. 
	
NUMA also measures the distance between the GPU and CPU devices. By default, each GPU selects a Numa CPU node that has the least NUMA distance between them; the host memory is automatically allocated closest to the memory pool of the NUMA node of the current GPU device. 
	
Note, using the *hipSetDevice* API with a different GPU provides access to the host allocation. However, it may have a longer NUMA distance.


### New Atomic System Scope Atomic Operations
	
HIP now provides new APIs with _system as a suffix to support system scope atomic operations. For example,  atomicAnd atomic is dedicated to the GPU device, and atomicAnd_system allows developers to extend the atomic operation to system scope from the GPU device to other CPUs and GPU devices in the system.
	
For more information, refer to the HIP Programming Guide at,
	
Add link

### Indirect Function Call and C++ Virtual Functions 
	
While the new release of the ROCm compiler supports indirect function calls and C++ virtual functions on a device, there are some known limitations and issues. 
	
**Limitations**
	
* An address to a function is device specific.  Note, a function address taken on the host can not be used on a device, and a function address taken on a device can not be used on the host.  On a system with multiple devices, an address taken on one device can not be used on a different device.
	
* C++ virtual functions only work on the device where the object was constructed.
	
* Indirect call to a device function with function scope shared memory allocation is not supported. For example, LDS.
	
* Indirect call to a device function defined in a source file different than the calling function/kernel is only supported when compiling the entire program with -fgpu-rdc.
	
**Known Issues in This Release**
	
* Programs containing kernels with different launch bounds may crash when making an indirect function call.  This issue is due to a compiler issue miscalculating the register budget for the callee function.
	
* Programs may not work correctly when making an indirect call to a function that uses more resources. For example, scratch memory, shared memory, registers made available by the caller.
	
* Compiling a program with objects with pure or deleted virtual functions on the device will result in a linker error.  This issue is due to the missing implementation of some C++ runtime functions on the device.
	
* Constructing an object with virtual functions in private or shared memory may crash the program due to a compiler issue when generating code for the constructor.  


## ROCm Data Center Tool 

### Prometheus (Grafana) Integration with Automatic Node Detection 

The ROCm Data Center (RDC) tool enables you to use Consul to discover the rdc_prometheus service automatically. Consul is “a service mesh solution providing a full-featured control plane with service discovery, configuration, and segmentation functionality.” For more information, refer to their website at https://www.consul.io/docs/intro.
	
The ROCm Data Center Tool uses Consul for health checks of RDC’s integration with the Prometheus plug-in (rdc_prometheus), and these checks provide information on its efficiency.  
	
Previously, when a new compute node was added, users had to change prometheus_targets.json to use Consul manually. Now, with the Consul agent integration, a new compute node can be discovered automatically.
	
Add link

### Coarse Grain Utilization
	
This feature provides a counter that displays the coarse grain GPU usage information, as shown below.
	
Sample output

```
	$ rocm_smi.py --showuse
	============================== % time GPU is busy =============================
               GPU[0] : GPU use (%): 0
               GPU[0] : GFX Activity: 3401
```

	
### Add 64-bit Energy Accumulator In-band
	
This feature provides an average value of energy consumed over time in a free-flowing RAPL counter, a 64-bit Energy Accumulator.
	
Sample output
	
```
	$ rocm_smi.py --showenergycounter
	=============================== Consumed Energy ================================
	GPU[0] : Energy counter: 2424868
	GPU[0] : Accumulated Energy (uJ): 0.0	

```	
	
### Support for Continuous Clocks Values
	
ROCm SMI will support continuous clock values instead of the previous discrete levels. Moving forward the updated sysfs file will consist of only MIN and MAX values and the user can set the clock value in the given range. 
	
Sample output:

```
	$ rocm_smi.py --setsrange 551 1270 
	Do you accept these terms? [y/N] y                                                                                    
	============================= Set Valid sclk Range=======
	GPU[0]          : Successfully set sclk from 551(MHz) to 1270(MHz)                                                     
	GPU[1]          : Successfully set sclk from 551(MHz) to 1270(MHz)                                                     
	=========================================================================
                       
	$ rocm_smi.py --showsclkrange                                                                                                                                                                    
	============================ Show Valid sclk Range======                     

	GPU[0]          : Valid sclk range: 551Mhz - 1270Mhz                                                                  
	GPU[1]          : Valid sclk range: 551Mhz - 1270Mhz             
```
	
## ROCm Math and Communication Libraries 

### rocBLAS

Enhancements and fixes:

* Added option to install script to build only rocBLAS clients with a pre-built rocBLAS library

* Supported gemm ext for unpacked int8 input layout on gfx908 GPUs

   * Added new flags rocblas_gemm_flags::rocblas_gemm_flags_pack_int8x4 to specify if using the packed layout

     * Set the rocblas_gemm_flags_pack_int8x4 when using packed int8x;, this should be always set on GPUs before gfx908

     * For gfx908 GPUs, unpacked int8 is supported. Setting of this flag is no longer required

     * Notice the default flags 0 uses unpacked int8 and changes the behaviour of int8 gemm from ROCm 4.1.0

* Added a query function rocblas_query_int8_layout_flag to get the preferable layout of int8 for gemm by device

For more information, refer to 

https://rocblas.readthedocs.io/en/master/


### rocRAND

* Performance fixes

For more information, refer to

https://rocrand.readthedocs.io/en/latest/


### rocSOLVER	

Support for:

* Multi-level logging functionality

* Implementation of the Thin-SVD algorithm

* Reductions of generalized symmetric- and hermitian-definite eigenproblems:

   * SYGS2, SYGST (with batched and strided_batched versions)
   * HEGS2, HEGST (with batched and strided_batched versions)

* Symmetric and hermitian matrix eigensolvers:

   * SYEV (with batched and strided_batched versions)
   * HEEV (with batched and strided_batched versions)
   
* Generalized symmetric- and hermitian-definite eigensolvers:

   * SYGV (with batched and strided_batched versions)
   * HEGV (with batched and strided_batched versions)

For more information, refer to

https://rocsolver.readthedocs.io/en/latest/


### rocSPARSE	

Enhancements:

* SpMM (CSR, COO)
* Code coverage analysis

For more information, refer to

https://rocsparse.readthedocs.io/en/latest/usermanual.html#rocsparse-gebsrmv


### hipSPARSE	

Enhancements:

* Generic API support, including SpMM (CSR, COO)
* csru2csr, csr2csru

For more information, refer to

https://rocsparse.readthedocs.io/en/latest/usermanual.html#types



# Fixed Defects

## Performance Impact for LDS-BOUND Kernels 

The following issue is fixed in the ROCm v4.2 release. 

The compiler in ROCm v4.1 generates LDS load and stores instructions that incorrectly assume equal performance between aligned and misaligned accesses. While this does not impact code correctness, it may result in sub-optimal performance.


# Known Issues 

The following are the known issues in this release.

## Upgrade to AMD ROCm v4.2 Not Supported

An upgrade from previous releases to AMD ROCm v4.2 is not supported. Complete uninstallation of previous ROCm versions is required before installing a new version of ROCm.

The hip-base package has a dependency on Perl modules that some operating systems may not have in their default package repositories.  Use the following commands to add repositories that have the required Perl packages:


#### For SLES 15 SP2

		sudo zypper addrepo 


For more information, see

https://download.opensuse.org/repositories/devel:languages:perl/SLE_15/devel:languages:perl.repo



#### For CentOS8.3

		sudo yum config-manager --set-enabled powertools
	

#### For RHEL8.3

		sudo subscription-manager repos --enable codeready-builder-for-rhel-8-x86_64-rpms
 
 
## Modulefile Fails to Install Automatically in ROCm Multi-Version Environment

The ROCm v4.2 release includes a preliminary implementation of environment modules to enable switching between multi versions of ROCm installation. The modulefile in */opt/rocm-4.2/lib/rocmmod* fails to install automatically in the ROCm multi-version environment.

This is a known limitation for environment modules in ROCm, and the issue is under investigation at this time. 

**Workaround**

Ensure you install the modulefile in */opt/rocm-4.2/lib/rocmmod* manually in a multi-version installation environment. 

For general information about modules, see
http://modules.sourceforge.net/ 

## Issue with Input/Output Types for Scan Algorithms in rocThrust

As rocThrust is updated to match CUDA Thrust 1.10, the different input/output types for scan algorithms in rocThrust/CUDA Thrust are no longer officially supported.  In this situation, the current C++ standard does not specify the intermediate accumulator type leading to potentially incorrect results and ill-defined behavior. 

As a workaround, users can:

* Use the same types for input and output

Or 

* For exclusive_scan, explicitly specify an *InitialValueType* in the last argument

Or 

* For inclusive_scan, which does not have an initial value argument, use a transform_iterator  to explicitly cast the input iterators to match the output’s value_type


## Precision Issue in AMD RADEON™ PRO VII and AMD RADEON™ VII

In AMD Radeon™ Pro VII AND AMD Radeon™ VII, a precision issue can occur when using the Tensorflow XLA path.

This issue is currently under investigation.



# Deprecations

This section describes deprecations and removals in AMD ROCm.

## Compiler Generated Code Object Version 2 Deprecation 

Compiler-generated code object version 2 is no longer supported and has been completely removed. Support for loading code object version 2 is also deprecated with no announced removal release.


# Deploying ROCm

AMD hosts both Debian and RPM repositories for the ROCm packages. 

For more information on ROCM installation on all platforms, see

https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html


# Machine Learning and High Performance Computing Software Stack for AMD GPU

For an updated version of the software stack for AMD GPU, see

https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html#software-stack-for-amd-gpu



# Hardware and Software Support
ROCm is focused on using AMD GPUs to accelerate computational tasks such as machine learning, engineering workloads, and scientific computing.
In order to focus our development efforts on these domains of interest, ROCm supports a targeted set of hardware configurations which are detailed further in this section.

**Note:** The AMD ROCm™ open software platform is a compute stack for headless system deployments. GUI-based software applications are currently not supported.

#### Supported GPUs
Because the ROCm Platform has a focus on particular computational domains, we offer official support for a selection of AMD GPUs that are designed to offer good performance and price in these domains.

**Note:** The integrated GPUs of Ryzen are not officially supported targets for ROCm.

ROCm officially supports AMD GPUs that use following chips:

* GFX9 GPUs

  - "Vega 10" chips, such as on the AMD Radeon RX Vega 64 and Radeon Instinct MI25
  
  - "Vega 7nm" chips, such as on the Radeon Instinct MI50, Radeon Instinct MI60 or AMD Radeon VII, Radeon Pro VII

* CDNA GPUs

  - MI100 chips such as on the AMD Instinct™ MI100


ROCm is a collection of software ranging from drivers and runtimes to libraries and developer tools.
Some of this software may work with more GPUs than the "officially supported" list above, though AMD does not make any official claims of support for these devices on the ROCm software platform.

The following list of GPUs are enabled in the ROCm software, though full support is not guaranteed:

  * GFX8 GPUs
    * "Polaris 11" chips, such as on the AMD Radeon RX 570 and Radeon Pro WX 4100
    * "Polaris 12" chips, such as on the AMD Radeon RX 550 and Radeon RX 540
  * GFX7 GPUs
    * "Hawaii" chips, such as the AMD Radeon R9 390X and FirePro W9100

As described in the next section, GFX8 GPUs require PCI Express 3.0 (PCIe 3.0) with support for PCIe atomics. This requires both CPU and motherboard support. GFX9 GPUs require PCIe 3.0 with support for PCIe atomics by default, but they can operate in most cases without this capability.

The integrated GPUs in AMD APUs are not officially supported targets for ROCm.
As described [below](#limited-support), "Carrizo", "Bristol Ridge", and "Raven Ridge" APUs are enabled in our upstream drivers and the ROCm OpenCL runtime.
However, they are not enabled in the HIP runtime, and may not work due to motherboard or OEM hardware limitations.
As such, they are not yet officially supported targets for ROCm.

For a more detailed list of hardware support, please see [the following documentation](https://en.wikipedia.org/wiki/List_of_AMD_graphics_processing_units).

#### Supported CPUs
As described above, GFX8 GPUs require PCIe 3.0 with PCIe atomics in order to run ROCm.
In particular, the CPU and every active PCIe point between the CPU and GPU require support for PCIe 3.0 and PCIe atomics.
The CPU root must indicate PCIe AtomicOp Completion capabilities and any intermediate switch must indicate PCIe AtomicOp Routing capabilities.

Current CPUs which support PCIe Gen3 + PCIe Atomics are:

  * AMD Ryzen CPUs
  * The CPUs in AMD Ryzen APUs
  * AMD Ryzen Threadripper CPUs
  * AMD EPYC CPUs
  * Intel Xeon E7 v3 or newer CPUs
  * Intel Xeon E5 v3 or newer CPUs
  * Intel Xeon E3 v3 or newer CPUs
  * Intel Core i7 v4, Core i5 v4, Core i3 v4 or newer CPUs (i.e. Haswell family or newer)
  * Some Ivy Bridge-E systems

Beginning with ROCm 1.8, GFX9 GPUs (such as Vega 10) no longer require PCIe atomics.
We have similarly opened up more options for number of PCIe lanes.
GFX9 GPUs can now be run on CPUs without PCIe atomics and on older PCIe generations, such as PCIe 2.0.
This is not supported on GPUs below GFX9, e.g. GFX8 cards in the Fiji and Polaris families.

If you are using any PCIe switches in your system, please note that PCIe Atomics are only supported on some switches, such as Broadcom PLX.
When you install your GPUs, make sure you install them in a PCIe 3.1.0 x16, x8, x4, or x1 slot attached either directly to the CPU's Root I/O controller or via a PCIe switch directly attached to the CPU's Root I/O controller.

In our experience, many issues stem from trying to use consumer motherboards which provide physical x16 connectors that are electrically connected as e.g. PCIe 2.0 x4, PCIe slots connected via the Southbridge PCIe I/O controller, or PCIe slots connected through a PCIe switch that does
not support PCIe atomics.

If you attempt to run ROCm on a system without proper PCIe atomic support, you may see an error in the kernel log (`dmesg`):
```
kfd: skipped device 1002:7300, PCI rejects atomics
```

Experimental support for our Hawaii (GFX7) GPUs (Radeon R9 290, R9 390, FirePro W9100, S9150, S9170)
does not require or take advantage of PCIe Atomics. However, we still recommend that you use a CPU
from the list provided above for compatibility purposes.

#### Not supported or limited support under ROCm

##### Limited support

* ROCm 4.x should support PCIe 2.0 enabled CPUs such as the AMD Opteron, Phenom, Phenom II, Athlon, Athlon X2, Athlon II and older Intel Xeon and Intel Core Architecture and Pentium CPUs. However, we have done very limited testing on these configurations, since our test farm has been catering to CPUs listed above. This is where we need community support. _If you find problems on such setups, please report these issues_.
* Thunderbolt 1, 2, and 3 enabled breakout boxes should now be able to work with ROCm. Thunderbolt 1 and 2 are PCIe 2.0 based, and thus are only supported with GPUs that do not require PCIe 3.1.0 atomics (e.g. Vega 10). However, we have done no testing on this configuration and would need community support due to limited access to this type of equipment.
* AMD "Carrizo" and "Bristol Ridge" APUs are enabled to run OpenCL, but do not yet support HIP or our libraries built on top of these compilers and runtimes.
  * As of ROCm 2.1, "Carrizo" and "Bristol Ridge" require the use of upstream kernel drivers.
  * In addition, various "Carrizo" and "Bristol Ridge" platforms may not work due to OEM and ODM choices when it comes to key configurations parameters such as inclusion of the required CRAT tables and IOMMU configuration parameters in the system BIOS.
  * Before purchasing such a system for ROCm, please verify that the BIOS provides an option for enabling IOMMUv2 and that the system BIOS properly exposes the correct CRAT table. Inquire with your vendor about the latter.
* AMD "Raven Ridge" APUs are enabled to run OpenCL, but do not yet support HIP or our libraries built on top of these compilers and runtimes.
  * As of ROCm 2.1, "Raven Ridge" requires the use of upstream kernel drivers.
  * In addition, various "Raven Ridge" platforms may not work due to OEM and ODM choices when it comes to key configurations parameters such as inclusion of the required CRAT tables and IOMMU configuration parameters in the system BIOS.
  * Before purchasing such a system for ROCm, please verify that the BIOS provides an option for enabling IOMMUv2 and that the system BIOS properly exposes the correct CRAT table. Inquire with your vendor about the latter.

##### Not supported

* "Tonga", "Iceland", "Vega M", and "Vega 12" GPUs are not supported.
* We do not support GFX8-class GPUs (Fiji, Polaris, etc.) on CPUs that do not have PCIe 3.0 with PCIe atomics.
  * As such, we do not support AMD Carrizo and Kaveri APUs as hosts for such GPUs.
  * Thunderbolt 1 and 2 enabled GPUs are not supported by GFX8 GPUs on ROCm. Thunderbolt 1 & 2 are based on PCIe 2.0.

In the default ROCm configuration, GFX8 and GFX9 GPUs require PCI Express 3.0 with PCIe atomics. The ROCm platform leverages these advanced capabilities to allow features such as user-level submission of work from the host to the GPU. This includes PCIe atomic Fetch and Add, Compare and Swap, Unconditional Swap, and AtomicOp Completion.

#### ROCm support in upstream Linux kernels

As of ROCm 1.9.0, the ROCm user-level software is compatible with the AMD drivers in certain upstream Linux kernels.
As such, users have the option of either using the ROCK kernel driver that are part of AMD's ROCm repositories or using the upstream driver and only installing ROCm user-level utilities from AMD's ROCm repositories.

These releases of the upstream Linux kernel support the following GPUs in ROCm:
 * 4.17: Fiji, Polaris 10, Polaris 11
 * 4.18: Fiji, Polaris 10, Polaris 11, Vega10
 * 4.20: Fiji, Polaris 10, Polaris 11, Vega10, Vega 7nm

The upstream driver may be useful for running ROCm software on systems that are not compatible with the kernel driver available in AMD's repositories.
For users that have the option of using either AMD's or the upstreamed driver, there are various tradeoffs to take into consideration:

|   | Using AMD's `rock-dkms` package | Using the upstream kernel driver |
| ---- | ------------------------------------------------------------| ----- |
| Pros | More GPU features, and they are enabled earlier | Includes the latest Linux kernel features |
|      | Tested by AMD on supported distributions | May work on other distributions and with custom kernels |
|      | Supported GPUs enabled regardless of kernel version | |
|      | Includes the latest GPU firmware | |
| Cons | May not work on all Linux distributions or versions | Features and hardware support varies depending on kernel version |
|      | Not currently supported on kernels newer than 5.4 | Limits GPU's usage of system memory to 3/8 of system memory (before 5.6). For 5.6 and beyond, both DKMS and upstream kernels allow use of 15/16 of system memory. |
|      |   | IPC and RDMA capabilities are not yet enabled |
|      |   | Not tested by AMD to the same level as `rock-dkms` package |
|      |   | Does not include most up-to-date firmware |


# Disclaimer

AMD®, the AMD Arrow logo, AMD Instinct™, Radeon™, ROCm® and combinations thereof are trademarks of Advanced Micro Devices, Inc.

Linux® is the registered trademark of Linus Torvalds in the U.S. and other countries.

PCIe® is a registered trademark of PCI-SIG Corporation. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.  

Google®  is a registered trademark of Google LLC.

Ubuntu and the Ubuntu logo are registered trademarks of Canonical Ltd.

Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.


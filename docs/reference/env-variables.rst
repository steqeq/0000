.. meta::
    :description: Environment variables reference
    :keywords: AMD, ROCm, environment variables, environment, reference

.. role:: cpp(code)
   :language: cpp

.. _env-variables-reference:

*************************************************************
ROCm environment variables
*************************************************************

The following table lists the most commonly used environment variables in the ROCm software stack. These variables help to perform simple tasks such as building a ROCm library or running applications on AMDGPUs.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Usage**

    * - | ``HIP_PATH``
        | The path of the HIP SDK on Microsoft Windows.
      - Default: ``C:/hip``

    * - | ``HIP_DIR``
        | The path of the HIP SDK on Microsoft Windows. This variable is ignored, if ``HIP_PATH`` is set.
      - Default: ``C:/hip``

    * - | ``ROCM_PATH``
        | The path of the installed ROCm software stack on Linux.
      - Default: ``/opt/rocm``

    * - | ``HIP_PLATFORM``
        | The platform targeted by HIP. If ``HIP_PLATFORM`` is not set, then HIPCC attempts to auto-detect the platform, if it can find NVCC.
      - ``amd``, ``nvidia``

CLR environment variables
=========================

AMD Common Language Runtime (CLR) library contains source codes for AMD's compute languages runtimes: HIP and OpenCL™. The environment variables affecting this library can also effect HIP and OpenCL™ libraries or applications. 
These environment variables are listed in the following table:

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Usage**

    * - | ``HIP_PLATFORM``
        | The platform targeted by HIP. If ``HIP_PLATFORM`` is not set, then HIPCC attempts to auto-detect the platform if it can find NVCC.
      - ``amd``, ``nvidia``

    * - | ``HSA_DISABLE_CACHE``
        | Disables the L2 cache.
      - | 0: Disable
        | 1: Enable

    * - | ``ROCM_HEADER_WRAPPER_WERROR``
        | Causes errors to be emitted instead of warnings.
      - | 0: Disable
        | 1: Enable

The following table lists the environment variables that affect the OpenCL and HIP implementation of CLR project.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Usage**

    * - | ``ROCM_LIBPATCH_VERSION``
        | The ROCm version in the integer format. The format is
        | :cpp:`MAJOR * 10000 + MINOR * 100 + PATCH`
      - 50000, 60020...

    * - | ``CPACK_DEBIAN_PACKAGE_RELEASE``
        | This is the numbering of the Debian package itself, i.e., the version of the packaging and not the version of the content.
      - 1, 2, 3...

    * - | ``CPACK_RPM_PACKAGE_RELEASE``
        | This is the numbering of the RPM package itself, i.e., the version of the packaging and not the version of the content.
      - 1, 2, 3...

The following table lists the environment variables that affect only the HIP implementation of the CLR project.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Usage**

    * - | ``HIP_FORCE_QUEUE_PROFILING``
        | Simulates the application to run in rocprof by forcing command queue profiling to ``on`` by default.
      - | 0: Disable
        | 1: Enable

    * - | ``HSA_OVERRIDE_GFX_VERSION``
        | Overrides the target version; used to enable HIP usage on unsupported hardware.
      - 11.0.0, 10.3.0

    * - | ``HSAKMT_DEBUG_LEVEL``
        | When set to the highest level, the system prints memory allocation information.
      - 1, 2, ... 7

The following table lists the environment variables affecting common runtime used in HIP and OpenCL (ROCclr) of clr project.

.. https://github.com/ROCm/clr/blob/develop/rocclr/utils/flags.hpp

.. list-table::
    :header-rows: 1
    :widths: 35,14,51

    * - **Environment variable**
      - **Default value**
      - **Usage**

    * - | ``AMD_CPU_AFFINITY``
        | Resets CPU affinity of any runtime threads
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``AMD_DIRECT_DISPATCH``
        | Enables direct kernel dispatch. Currently available on Linux; under development for Windows.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``AMD_GPU_FORCE_SINGLE_FP_DENORM``
        | Force denorm for single precision.
      - ``-1``
      - | -1: Don't force 
        | 0: Disable
        | 1: Enable

    * - | ``AMD_LOG_LEVEL``
        | Enables HIP log on various level.
      - ``0``
      - | 0: Disable log. 
        | 1: Enables log on error level.
        | 2: Enable log on warning and below levels.
        | 3: Enable log on information and below levels.
        | 4: Decodes and displays the AQL packets.

    * - | ``AMD_LOG_LEVEL_FILE``
        | Sets output file for ``AMD_LOG_LEVEL``.
      - stderr output
      - 

    * - | ``AMD_LOG_MASK``
        | Enable HIP log on different level.
      - ``0x7FFFFFFF``
      - | 0x1: Log API calls.
        | 0x02: Kernel and Copy Commands and Barriers.
        | 0x4: Synchronization and waiting for commands to finish.
        | 0x8: Enable log on information and below levels.
        | 0x20: Queue commands and queue contents.
        | 0x40: Signal creation, allocation, pool.
        | 0x80: Locks and thread-safety code.
        | 0x100: Copy debug.
        | 0x200: Detailed copy debug.
        | 0x400: Resource allocation, performance-impacting events.
        | 0x800: Initialization and shutdown.
        | 0x1000: Misc debug, not yet classified.
        | 0x2000: Show raw bytes of AQL packet.
        | 0x4000: Show code creation debug.
        | 0x8000: More detailed command info, including barrier commands.
        | 0x10000: Log message location.
        | 0xFFFFFFFF: Log always even mask flag is zero.

    * - | ``AMD_OCL_BUILD_OPTIONS``
        | Set the options for clBuildProgram and clCompileProgram, override.
      - Unset
      - 

    * - | ``AMD_OCL_BUILD_OPTIONS_APPEND``
        | Appends the options for ``clBuildProgram`` and ``clCompileProgram``.
      - Unset
      - 

    * - | ``AMD_OCL_LINK_OPTIONS``
        | Sets the options for ``clLinkProgram``.
      - Unset
      - 

    * - | ``AMD_OCL_LINK_OPTIONS_APPEND``
        | Appends the options for ``clLinkProgram``.
      - Unset
      - 

    * - | ``AMD_OCL_WAIT_COMMAND``
        | Enable a wait for every submitted command.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``OCL_SET_SVM_SIZE``
        | Sets Shared Virtual Memory (SVM) space size (in Byte) for discrete GPUs.
      - ``65536``
      -

    * - | ``OCL_STUB_PROGRAMS``
        | Enables OCL programs stubing.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``OPENCL_VERSION``
        | Force GPU OpenCL version.
      - ``200``
      - 

    * - | ``AMD_OPT_FLUSH``
        | Sets kernel flush option.
      - ``0x1``
      - | ``0x0`` = Uses system-scope fence operations.
        | ``0x1`` = Uses device-scope fence operations when possible.

    * - | ``AMD_SERIALIZE_COPY``
        | Controls serialization of copies
      - ``0``
      - | 0: Disable
        | 1: Wait for completion before enqueue.
        | 2: Wait for completion after enqueue.
        | 3: Both

    * - | ``AMD_SERIALIZE_KERNEL``
        | Serialize kernel enqueue.
      - ``0``
      - | 0: Disable
        | 1: Wait for completion before enqueue.
        | 2: Wait for completion after enqueue.
        | 3: Both

    * - | ``AMD_THREAD_TRACE_ENABLE``
        | Enables thread trace extension.
      - ``1``
      - | 0: Disable
        | 1: Enable

    * - | ``CL_KHR_FP64``
        | Controls support for double precision.
      - ``1``
      - | 0: Disable
        | 1: Enable

    * - | ``CQ_THREAD_STACK_SIZE``
        | The default command queue thread stack size in Bytes.
      - ``262144``
      - The default value corresponds to 256 KB. 

    * - | ``CUDA_VISIBLE_DEVICES``
        | The visible devices to HIP (whose indices are present in the sequence)
      - Unset
      - ``0,1,2``: List of the device indices. Depending on the number of devices in the system.

    * - | ``DEBUG_CLR_GRAPH_PACKET_CAPTURE``
        | Controls capturing of graph packets.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``DEBUG_CLR_LIMIT_BLIT_WG``
        | Sets the limit for the number of workgroups in blit operations.
      - ``16``
      -

    * - | ``DISABLE_DEFERRED_ALLOC``
        | Controls deferred memory allocation on device.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``GPU_ADD_HBCC_SIZE``
        | Adds HBCC size to the reported device memory.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``GPU_ANALYZE_HANG``
        | Allows you to analyze GPU hang issue.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``GPU_BLIT_ENGINE_TYPE``
        | Specifies blit engine type.
      - ``0``
      - | 0: Default
        | 1: Host
        | 2: CAL
        | 3: Kernel

    * - | ``GPU_CP_DMA_COPY_SIZE``
        | Set maximum size of CP DMA copy in kB.
      - ``1``
      -

    * - | ``GPU_DEBUG_ENABLE``
        | Enables collection of extra information for debugger at the cost of performance.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``GPU_DEVICE_ORDINAL``
        | Selects the device ordinal, which is a comma separated list of available devices.
      - Unset
      - A value of ``0,2`` exposes devices 1 and 3 in the system.

    * - | ``GPU_DUMP_BLIT_KERNELS``
        | Controls dumping of the kernels for blit manager.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``GPU_DUMP_CODE_OBJECT``
        | Controls dumping of code object.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``GPU_ENABLE_COOP_GROUPS``
        | Enables cooperative group launch.
      - ``1``
      - | 0: Disable
        | 1: Enable

    * - | ``GPU_ENABLE_HW_P2P``
        | Enables hardware peer to peer (P2P) path.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``GPU_ENABLE_LC``
        | Enables LC path.
      - ``1``
      - | 0: Disable
        | 1: Enable

    * - | ``GPU_ENABLE_PAL``
        | Specifies PAL backend.
      - ``2``
      - | 0: ROC
        | 1: PAL
        | 2: ROC or PAL

    * - | ``GPU_ENABLE_WAVE32_MODE``
        | Enables Wave32 compilation in hardware, if available.
      - ``1``
      - | 0: Disable
        | 1: Enable

    * - | ``GPU_ENABLE_WGP_MODE``
        | Enables WGP Mode in hardware, if available. Workgroups of waves are
        | dispatched in one of two modes: CU or WGP.
      - ``1``
      - | 0: CU mode. The waves of a workgroup are distributed across just two SIMD32’s.
        | 1: WGP mode. The waves of a workgroup are distributed across all 4 SIMD32’s within a workgroup.

    * - | ``GPU_FORCE_BLIT_COPY_SIZE``
        | Specifies the threshold size in KB, under which blit is forced instead of SDMA.
      - 0
      -

    * - | ``GPU_FORCE_QUEUE_PROFILING``
        | Forces command queue profiling.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``GPU_FLUSH_ON_EXECUTION``
        | Submits commands to hardware on every operation.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``GPU_IMAGE_BUFFER_WAR``
        | Enables image buffer workaround.
      - ``1``
      - | 0: Disable
        | 1: Enable

    * - | ``GPU_IMAGE_DMA``
        | Enable DRM DMA for image transfers.
      - ``1``
      - | 0: Disable
        | 1: Enable

    * - | ``GPU_MAX_COMMAND_BUFFERS``
        | Sets the maximum number of command buffers allocated per queue.
      - ``8``
      -

    * - | ``GPU_MAX_HEAP_SIZE``
        | Sets the maximum size of the GPU heap (in percentage) on the board memory.
      - ``100``
      -

    * - | ``GPU_MAX_HW_QUEUES``
        | Sets the maximum number of hardware queues to be allocated per device.
      - ``4``
      - The variable controls how many independent hardware queues HIP runtime can create per process,
        per device. If an application allocates more HIP streams than this number, then HIP runtime reuses
        the same hardware queues for the new streams in a round-robin manner. Note that this maximum
        number does not apply to hardware queues that are created for CU-masked HIP streams, or
        cooperative queues for HIP Cooperative Groups (single queue per device).

    * - | ``GPU_MAX_REMOTE_MEM_SIZE``
        | Maximum size that allows device memory substitution with system.
      - ``2``
      -

    * - | ``GPU_MAX_SUBALLOC_SIZE``
        | Sets the maximum size for sub-allocations in KB.
      - ``4096``
      -

    * - | ``GPU_MAX_USWC_ALLOC_SIZE``
        | Sets  the maximum USWC allocation size in MB.
      - ``2048``
      - -1: No limit

    * - | ``GPU_MAX_WORKGROUP_SIZE``
        | Sets the maximum number of workitems in a workgroup for GPU.
      - ``0``
      - 0: Ignore the environment variable and use the default workgroup size, which is 256.

    * - | ``GPU_MIPMAP``
        | Enables GPU mipmap extension.
      - ``1``
      - | 0: Disable
        | 1: Enable

    * - | ``GPU_NUM_COMPUTE_RINGS``
        | Sets the number of GPU compute rings.
      - ``2``
      - | 0: Disable
        | 1,2, etc. ...: Number of compute rings

    * - | ``GPU_NUM_MEM_DEPENDENCY``
        | Sets the number of memory objects for dependency tracking.
      - ``256``
      -

    * - | ``GPU_PINNED_MIN_XFER_SIZE``
        | Sets the minimum buffer size (in MB) for pinned read and write transfers.
      - ``128``
      -

    * - | ``GPU_PINNED_XFER_SIZE``
        | Sets the buffer size (in MB) for pinned read and write transfers.
      - ``32``
      -

    * - | ``GPU_PRINT_CHILD_KERNEL``
        | Specifies the number of child kernels to be printed.
      - ``0``
      -

    * - | ``GPU_RESOURCE_CACHE_SIZE``
        | Sets the resource cache size in MB.
      - ``64``
      -

    * - | ``GPU_SINGLE_ALLOC_PERCENT``
        | Sets the maximum size of a single allocation as a percentage of  the total.
      - ``85``
      - 

    * - | ``GPU_STAGING_BUFFER_SIZE``
        | Sets the GPU staging buffer size in MB.
      - ``4``
      -

    * - | ``GPU_STREAMOPS_CP_WAIT``
        | Force the stream memory operation to wait on CP.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``GPU_USE_DEVICE_QUEUE``
        | Controls use of dedicated device queue for the actual submissions.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``GPU_WAVES_PER_SIMD``
        | Force the number of waves per SIMD. 1-10
      - ``0``
      - 

    * - | ``GPU_XFER_BUFFER_SIZE``
        | Transfer buffer size for image copy optimization in KB.
      - ``0``
      -
        
    * - | ``HIP_FORCE_DEV_KERNARG``
        | Force device memory for kernel args.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``HIP_HIDDEN_FREE_MEM``
        | Amount of memory to hide from the free memory reported by hipMemGetInfo.
      - ``0``
      - 0: Disable

    * - | ``HIP_HOST_COHERENT``
        | Coherent memory in ``hipHostMalloc``.
      - ``0``
      - | 0: Memory is not coherent between host and GPU.
        | 1: Memory is coherent with host.
        | Environment variable has effect, if:
        | - One of the HostMalloc flags is set.
        | - ``hipHostMallocCoherent=0``
        | - ``hipHostMallocNonCoherent=0``
        | - ``hipHostMallocMapped=0``

    * - | ``HIP_INITIAL_DM_SIZE``
        | Sets the initial heap size for device malloc.
      - ``8388608``
      - The default value corresponds to 8 MB. 

    * - | ``HIP_LAUNCH_BLOCKING``
        | Controls serialization of kernel execution.
      - ``0``
      - | 0: Disable. Kernel executes normally.
        | 1: Enable. Serializes kernel enqueue, behaves the same as ``AMD_SERIALIZE_KERNEL``.

    * - | ``HIP_MEM_POOL_SUPPORT``
        | Enables memory pool support in HIP.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``HIP_MEM_POOL_USE_VM``
        | Enables memory pool support in HIP.
      - | ``0``: other OS
        | ``1``: Windows
      - | 0: Disable
        | 1: Enable

    * - | ``HIP_USE_RUNTIME_UNBUNDLER``
        | Controls use of runtime code object unbundler.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``HIP_VISIBLE_DEVICES``
        | Specifies the indices of the devices allowed to be visible to HIP.
      - None
      - 0,1,2: Depending on the number of devices on the system.

    * - | ``HIP_VMEM_MANAGE_SUPPORT``
        | Enables virtual memory management support.
      - ``1``
      - | 0: Disable
        | 1: Enable

    * - | ``HIPCC_VERBOSE``
        | Controls the extra information to be displayed during the build such as compiler flags, paths etc.
      - ``0``
      - 

    * - | ``HIPRTC_COMPILE_OPTIONS_APPEND``
        | Sets compile options needed for ``hiprtc`` compilation.
      - None
      - 

    * - | ``HIPRTC_LINK_OPTIONS_APPEND``
        | Sets link options needed for ``hiprtc`` compilation.
      - None
      - 

    * - | ``HIPRTC_USE_RUNTIME_UNBUNDLER``
        | Set this to ``true`` to force runtime unbundler in hiprtc.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``HSA_KERNARG_POOL_SIZE``
        | Sets the pool size for kernel arguments.
      - ``1048576``
      - The default value corresponds to 1 megabyte (MB).

    * - | ``HSA_LOCAL_MEMORY_ENABLE``
        | Enables use of local memory on HSA device.
      - ``1``
      - | 0: Disable
        | 1: Enable

    * - | ``PAL_DISABLE_SDMA``
        | Disable SDMA for PAL.
      - ``0``
      - | 0: Enable SDMA for PAL.
        | 1: Disable SDMA for PAL.

    * - | ``PAL_MALL_POLICY``
        | Controls the behaviour of allocations with respect to the MALL.
      - ``0``
      - | 0: MALL policy is decided by KMD.
        | 1: Allocations are never put through the MALL.
        | 2: Allocations will always be put through the MALL.

    * - | ``PAL_ALWAYS_RESIDENT``
        | Forces memory resources to become resident during allocation.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``PAL_EMBED_KERNEL_MD``
        | Enables writing kernel metadata into command buffers.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``PAL_FORCE_ASIC_REVISION``
        | Forces a specific ASIC revision on all devices.
      - ``0``
      -

    * - | ``PAL_HIP_IPC_FLAG``
        | Enables inter-process flag for device allocation in PAL HIP.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``PAL_PREPINNED_MEMORY_SIZE``
        | Sets the size of pre-pinned memory.
      - ``64``
      -

    * - | ``PAL_RGP_DISP_COUNT``
        | Sets the number of dispatches for RGP capture with SQTT.
      - ``10000``
      -

    * - | ``REMOTE_ALLOC``
        | Enables use of remote memory for the global heap allocation.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``ROC_ACTIVE_WAIT_TIMEOUT``
        | Forces active wait of GPU interrupt for the timeout in us.
      - ``0``
      -

    * - | ``ROC_AQL_QUEUE_SIZE``
        | AQL queue size in the AQL packets in Bytes.
      - ``16384``
      - The default value corresponds to 16 KB.

    * - | ``ROC_CPU_WAIT_FOR_SIGNAL``
        | Enable CPU wait for dependent HSA signals.
      - ``1``
      - | 0: Disable
        | 1: Enable

    * - | ``ROC_ENABLE_LARGE_BAR``
        | Enable Large Bar if supported by the device.
      - ``1``
      - | 0: Disable
        | 1: Enable

    * - | ``ROC_GLOBAL_CU_MASK``
        | Sets a global CU mask, entered as hex value for all queues. Each active bit represents one CU, e.g., ``0xf`` enables 4 CUs.
      - None
      - 

    * - | ``ROC_HMM_FLAGS``
        | Sets ROCm HMM configuration flags.
      - ``0``
      - 

    * - | ``ROC_P2P_SDMA_SIZE``
        | The minimum size in KB for peer to peer (P2P) transfer with SDMA.
      - ``1024``
      - The default value corresponds to 1 megabyte (MB).

    * - | ``ROC_SIGNAL_POOL_SIZE``
        | Sets the initial size for HSA signal pool.
      - ``32``
      - 

    * - | ``ROC_SKIP_KERNEL_ARG_COPY``
        | If ``true``, then runtime can skip kernel arg copy.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``ROC_SYSTEM_SCOPE_SIGNAL``
        | Enable system scope for signals, uses interrupts.
      - ``1``
      - | 0: Disable
        | 1: Enable

    * - | ``ROC_USE_FGS_KERNARG``
        | Use fine grain kernel arguments segment for supported ASICs.
      - ``1``
      - | 0: Disable
        | 1: Enable

    * - | ``ROCPROFILER_REGISTER_ROOT``
        | Sets the path to ``rocProfiler``.
      - None
      - 

The following table lists debug environment variables affecting common runtime used in HIP and OpenCL (ROCclr) of clr project. These variables only settable during DEBUG build.

.. list-table::
    :header-rows: 1
    :widths: 35,14,51

    * - **Environment variable**
      - **Default value**
      - **Usage**

    * - | ``AMD_OCL_SUBST_OBJFILE``
        | Specifies binary substitution config file for OpenCL.
      - None
      - 

    * - | ``CPU_MEMORY_ALIGNMENT_SIZE``
        | Size in bytes for the default alignment for guarded memory on CPU.
      - 256
      -

    * - | ``CPU_MEMORY_GUARD_PAGE_SIZE``
        | Size of the CPU memory guard page in KB.
      - ``64``
      - The default value corresponds to 64 KB. 

    * - | ``CPU_MEMORY_GUARD_PAGES``
        | Enables using guard pages for CPU memory.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``MEMOBJ_BASE_ADDR_ALIGN``
        | Alignment of the base address of any allocate memory object.
      - ``4096``
      - The default value corresponds to 4 KB. 

    * - | ``PARAMETERS_MIN_ALIGNMENT``
        | Specifies the minimum alignment required for the abstract parameters stack.
      - 64 at ``__AVX512F__``, 32 at ``__AVX__`` and 16 at other cases
      -

ROCR-Runtime Environment Variables
==================================

.. https://github.com/ROCm/ROCR-Runtime/blob/master/src/core/util/flag.h
.. We need to extend the following list.

AMD ROCR-Runtime environment variables:

.. list-table::
    :header-rows: 1
    :widths: 35,14,51

    * - **Environment variable**
      - **Default value**
      - **Usage**

    * - | ``ROCR_VISIBLE_DEVICES``
        | A list of device indices or UUIDs that will be exposed to applications.
      - None
      - ``0,GPU-DEADBEEFDEADBEEF``

    * - | ``HSA_SCRATCH_MEM``
        | Specifies the maximum amount of scratch memory that can be used per process per GPU.
      -
      -

    * - | ``HSA_XNACK``
        | Turning on XNACK by setting the environment variable.
      - None
      - ``1``

    * - | ``HSA_CU_MASK``
        | Sets the mask on a lower level of queue creation in the driver. 
        | This mask is also applied to the queues being profiled.
      - None
      - ``1:0-8``

    * - | ``HSA_ENABLE_SDMA``
        | Enables the use of DMA engines in all copy directions (Host-to-Device, Device-to-Host, Device-to-Device), when using any of the following APIs:
        | ``hsa_memory_copy``, 
        | ``hsa_amd_memory_fill``, 
        | ``hsa_amd_memory_async_copy``, 
        | ``hsa_amd_memory_async_copy_on_engine``.
      - ``1``
      - | 0: Disable
        | 1: Enable

    * - | ``HSA_ENABLE_PEER_SDMA``
        | Enables the use of DMA engines for Device-to-Device copies, when using any of the following APIs:
        | ``hsa_memory_copy``,
        | ``hsa_amd_memory_async_copy``,
        | ``hsa_amd_memory_async_copy_on_engine``.
      - ``1``
      - | 0: Disable
        | 1: Enable
        |
        | The value of ``HSA_ENABLE_PEER_SDMA`` is ignored,
        | if ``HSA_ENABLE_SDMA`` set to ``0``.

rocPRIM environment variables
=============================

The following table lists the environment variables used in the rocPRIM library.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Default value**

    * - | ``HIP_PATH``
        | Specifies the path of the HIP SDK on Microsoft Windows.
      - ``C:/hip``

    * - | ``HIP_DIR``
        | The path of the HIP SDK on Microsoft Windows. This variable is ignored, if ``HIP_PATH`` is set.
      - ``C:/hip``

    * - | ``VCPKG_PATH``
        | Specifies the path of the ``vcpkg`` package manager on Microsoft Windows. This environment variable has no effect on Linux.
      - ``C:/github/vcpkg``

    * - | ``ROCM_PATH``
        | Specifies the path of the installed ROCm software stack on Linux.
      - ``/opt/rocm``

    * - | ``ROCM_CMAKE_PATH``
        | Specifies the path of the installed ROCm ``cmake`` file on Microsoft Windows.
      - ``C:/hipSDK``

    * - | ``HIPCC_COMPILE_FLAGS_APPEND``
        | Enables extra ``amdclang++`` compiler flags on Linux. This environment variable is ignored if ``CXX`` environment variable is set.
      - None

    * - | ``ROCPRIM_USE_HMM``
        | The tests suite uses unified memory, if it's set to 1 during the tests
        | run.
      - Unset

    * - | ``CTEST_RESOURCE_GROUP_0``
        | Used by CI, and helps to group the tests for different CI steps. Most
        | users should ignore this.
      - Unset

hipCUB environment variables
============================

Environment variables of hipCUB library.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Usage**

    * - | ``HIP_PATH``
        | The path of the HIP SDK on Microsoft Windows.
      - Default: ``C:/hip``

    * - | ``HIP_DIR``
        | The path of the HIP SDK on Microsoft Windows. This variable is ignored, if ``HIP_PATH`` is set.
      - Default: ``C:/hip``

    * - | ``VCPKG_PATH``
        | The path of the vcpkg package manager on Microsoft Windows. On Linux 
        | this environment variable has no effect.
      - ``C:/github/vcpkg``

    * - | ``ROCM_PATH``
        | The path of the installed ROCm software stack on Linux.
      - Default: ``/opt/rocm``

    * - | ``HIPCC_COMPILE_FLAGS_APPEND``
        | Extra amdclang or amdclang++ compiler flags on Linux. 
        | amdclang++ ignores this, if CXX environment variable is set.
        | amdclang ignores this, if CC environment variable is set.
      - Unset

    * - | ``HIPCUB_USE_HMM``
        | The tests suite uses unified memory, if it's set to 1 during the tests
        | run.
      - Unset

    * - | ``CTEST_RESOURCE_GROUP_0``
        | Used by CI, and helps to group the tests for different CI steps. Most
        | users should ignore this.
      - Unset

rocThrust environment variables
===============================

The following table lists the environment variables used in the rocThrust library.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Usage**

    * - | ``HIP_PATH``
        | The path of the HIP SDK on Microsoft Windows.
      - Default: ``C:/hip``

    * - | ``HIP_DIR``
        | The path of the HIP SDK on Microsoft Windows. This variable is ignored, if ``HIP_PATH`` is set.
      - Default: ``C:/hip``

    * - | ``VCPKG_PATH``
        | The path of the vcpkg package manager on Microsoft Windows. On Linux 
        | this environment variable has no effect.
      - ``C:/github/vcpkg``

    * - | ``ROCM_PATH``
        | The path of the installed ROCm software stack on Linux.
      - Default: ``/opt/rocm``

    * - | ``ROCTHRUST_USE_HMM``
        | Sets the tests to use the unified memory allocation during tests run.
      - Unset

    * - | ``CTEST_RESOURCE_GROUP_0``
        | Used by CI, and helps to group the tests for different CI steps. Most
        | users should ignore this.
      - Unset

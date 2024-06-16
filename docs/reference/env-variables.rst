.. meta::
    :description: Environment variables reference
    :keywords: AMD, ROCm, environment variables, environment, reference

.. role:: cpp(code)
   :language: cpp

.. _env-variables-reference:

*************************************************************
Environment variables reference
*************************************************************

ROCm common and important Environment Variables
===============================================

The following table contains the environment variables that are most commonly used in the ROCm software stack. These variables can be useful for simple tasks, like building a ROCm library or running applications on AMD cards.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Usage**

    * - | ``HIP_DIR``
        | The path of the HIP SDK on Microsoft Windows.
      - Default: ``C:/hip``

    * - | ``HIP_PATH``
        | The path of the HIP SDK on Microsoft Windows. On linux the ``ROCM_PATH``
        | environment variable used to set different ROCm root path.
      - Default: ``C:/hip``

    * - | ``ROCM_PATH``
        | The path of the installed ROCm software stack on linux. On Microsoft 
        | Windows the ``HIP_DIR`` environment variable used to set 
        | different HIP SDK path.
      - Default: ``/opt/rocm``

    * - | ``HIP_PLATFORM``
        | The platform HIP is targeting. If ``HIP_PLATFORM`` is not set, then hipcc
        | will attempt to auto-detect based on if nvcc is found.
      - ``amd``, ``nvidia``

CLR Environment Variables
=========================

AMD Common Language Runtime (clr) contains source codes for AMD's compute languages runtimes: **HIP** and **OpenCL™**. The environment variables, which are effecting this library can effect **HIP** and **OpenCL™** libraries or applications too.

The following table contains the environment variables effecting all backends of project clr.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Usage**

    * - | ``HIP_PLATFORM``
        | The platform HIP is targeting. If ``HIP_PLATFORM`` is not set, then hipcc will attempt to auto-detect based on if nvcc is found.
      - ``amd``, ``nvidia``

    * - | ``HSA_DISABLE_CACHE``
        | Used to disable L2 cache.
      - | 0: Disable
        | 1: Enable

    * - | ``ROCM_HEADER_WRAPPER_WERROR``
        | Causes errors to be emitted instead of warnings.
      - | 0: Disable
        | 1: Enable

The following table contains the environment variables effecting the opencl and hipamd backend of project clr.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Usage**

    * - | ``ROCM_LIBPATCH_VERSION``
        | The ROCm version in the format of an integer. The format is
        | :cpp:`MAJOR * 10000 + MINOR * 100 + PATCH`
      - 50000, 60020...

    * - | ``CPACK_DEBIAN_PACKAGE_RELEASE``
        | This is the numbering of the Debian package itself, i.e. the version of the packaging and not the version of the content.
      - 1, 2, 3...

    * - | ``CPACK_RPM_PACKAGE_RELEASE``
        | This is the numbering of the RPM package itself, i.e. the version of the packaging and not the version of the content.
      - 1, 2, 3...

The following table contains the environment variables effecting the hipamd backend of project clr.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Usage**

    * - | ``HIP_FORCE_QUEUE_PROFILING``
        | Used to run the app as if it were run in rocprof. Forces command queue profiling on by default.
      - | 0: Disable
        | 1: Enable

    * - | ``HSA_OVERRIDE_GFX_VERSION``
        | Override the target version. Used to enable HIP usage on unsupported hardware.
      - 11.0.0, 10.3.0

    * - | ``HSAKMT_DEBUG_LEVEL``
        | When set to the highest level, the system will print memory allocation info.
      - 1, 2, ... 7

rocclr Environment Variables
----------------------------

AMD rocclr environment variables at release build:

.. https://github.com/ROCm/clr/blob/develop/rocclr/utils/flags.hpp

.. list-table::
    :header-rows: 1
    :widths: 35,14,51

    * - **Environment variable**
      - **Default value**
      - **Usage**

    * - | ``AMD_CPU_AFFINITY``
        | Reset CPU affinity of any runtime threads
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``AMD_DIRECT_DISPATCH``
        | Enable direct kernel dispatch (Currently for Linux; under development for Windows).
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``AMD_GPU_FORCE_SINGLE_FP_DENORM``
        | Force denorm for single precision: -1 - don't force, 0 - disable, 1 - enable
      - ``-1``
      - | -1: Don't force 
        | 0: Disable
        | 1: Enable

    * - | ``AMD_LOG_LEVEL``
        | Enable HIP log on different Level.
      - ``0``
      - | 0: Disable log. 
        | 1: Enable log on error level.
        | 2: Enable log on warning and below levels.
        | 0x3: Enable log on information and below levels.
        | 0x4: Decode and display AQL packets.

    * - | ``AMD_LOG_LEVEL_FILE``
        | Set output file for AMD_LOG_LEVEL.
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
      - Unset by default.
      - 

    * - | ``AMD_OCL_BUILD_OPTIONS_APPEND``
        | Append the options for clBuildProgram and clCompileProgram.
      - Unset by default.
      - 

    * - | ``AMD_OCL_LINK_OPTIONS``
        | Set the options for clLinkProgram, override.
      - Unset by default.
      - 

    * - | ``AMD_OCL_LINK_OPTIONS_APPEND``
        | Append the options for clLinkProgram.
      - Unset by default.
      - 

    * - | ``AMD_OCL_WAIT_COMMAND``
        | Enable a wait for every submitted command.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``OCL_SET_SVM_SIZE``
        | Set Shared Virtual Memory (SVM) space size for discrete GPU.
      - ``65536``
      - Unit: Byte

    * - | ``OCL_STUB_PROGRAMS``
        | Enables OCL programs stubing.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``OPENCL_VERSION``
        | Force GPU opencl version.
      - ``200``
      - 

    * - | ``AMD_OPT_FLUSH``
        | Kernel flush option.
      - ``0x1``
      - | ``0x0`` = Use system-scope fence operations.
        | ``0x1`` = Use device-scope fence operations when possible.

    * - | ``AMD_SERIALIZE_COPY``
        | Serialize copies
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
        | Enable thread trace extension.
      - ``1``
      - | 0: Disable
        | 1: Enable

    * - | ``CL_KHR_FP64``
        | Enable/Disable support for double precision.
      - ``1``
      - | 0: Disable
        | 1: Enable

    * - | ``CQ_THREAD_STACK_SIZE``
        | The default command queue thread stack size.
      - ``262144``
      - | Unit: Byte 
        | The default value corresponds to  256  kilobyte (kB). 

    * - | ``CUDA_VISIBLE_DEVICES``
        | Only devices whose index is present in the sequence are visible to HIP
      - Unset by default.
      - ``0,1,2``: Depending on the number of devices on the system.

    * - | ``DEBUG_CLR_GRAPH_PACKET_CAPTURE``
        | Enable/Disable graph packet capturing.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``DEBUG_CLR_LIMIT_BLIT_WG``
        | Limit the number of workgroups in blit operations.
      - 16
      -

    * - | ``DISABLE_DEFERRED_ALLOC``
        | Disables deferred memory allocation on device.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``GPU_ADD_HBCC_SIZE``
        | Add HBCC size to the reported device memory.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``GPU_ANALYZE_HANG``
        | 1 = Enables GPU hang analysis
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``GPU_BLIT_ENGINE_TYPE``
        | Blit engine type.
      - ``0``
      - | 0: Default
        | 1: Host
        | 2: CAL
        | 3: Kernel

    * - | ``GPU_CP_DMA_COPY_SIZE``
        | Set maximum size of CP DMA copy in kB.
      - ``1``
      - Unit: kilobyte (kB)

    * - | ``GPU_DEBUG_ENABLE``
        | Enables collection of extra info for debugger at some performance cost.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``GPU_DEVICE_ORDINAL``
        | Select the device ordinal, a comma separated list of available devices.
      - Unset by default.
      - ``0,2``: Expose the 1. and 3. device in the system.

    * - | ``GPU_DUMP_BLIT_KERNELS``
        | Dump the kernels for blit manager.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``GPU_DUMP_CODE_OBJECT``
        | Dump code object.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``GPU_ENABLE_COOP_GROUPS``
        | Enables cooperative group launch.
      - ``1``
      - | 0: Disable
        | 1: Enable

    * - | ``GPU_ENABLE_HW_P2P``
        | Enables HW P2P path.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``GPU_ENABLE_LC``
        | Enables LC path.
      - ``1``
      - | 0: Disable
        | 1: Enable

    * - | ``GPU_ENABLE_PAL``
        | Enables PAL backend.
      - ``2``
      - | 0: ROC
        | 1: PAL
        | 2: ROC or PAL

    * - | ``GPU_ENABLE_WAVE32_MODE``
        | Enables Wave32 compilation in HW if available.
      - ``1``
      - | 0: Disable
        | 1: Enable

    * - | ``GPU_ENABLE_WGP_MODE``
        | Enables WGP Mode in HW if available.
      - ``1``
      - | 0: Disable
        | 1: Enable

    * - | ``GPU_FORCE_BLIT_COPY_SIZE``
        | Size in KB of the threshold below which to force blit instead for sdma.
      - 0
      - Unit: kilobyte (kB)

    * - | ``GPU_FORCE_QUEUE_PROFILING``
        | Force command queue profiling by default.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``GPU_FLUSH_ON_EXECUTION``
        | Submit commands to HW on every operation.
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
        | The maximum number of command buffers allocated per queue.
      - ``8``
      -

    * - | ``GPU_MAX_HEAP_SIZE``
        | Set maximum size of the GPU heap to % of board memory.
      - ``100``
      - | Unit: Percentage 

    * - | ``GPU_MAX_HW_QUEUES``
        | The maximum number of hardware queues allocated per device.
      - ``4``
      - The variable controls how many independent hardware queues HIP runtime can create per process,
        per device. If an application allocates more HIP streams than this number, then HIP runtime reuses
        the same hardware queues for the new streams in a round-robin manner. Note that this maximum
        number does not apply to hardware queues that are created for CU-masked HIP streams, or
        cooperative queues for HIP Cooperative Groups (single queue per device).

    * - | ``GPU_MAX_REMOTE_MEM_SIZE``
        | Maximum size that allows device memory substitution with system.
      - ``2``
      - Unit: kilobyte (kB)

    * - | ``GPU_MAX_SUBALLOC_SIZE``
        | The maximum size accepted for suballocaitons in KB.
      - ``4096``
      - Unit: kilobyte (kB)

    * - | ``GPU_MAX_USWC_ALLOC_SIZE``
        | Set a limit in Mb on the maximum USWC allocation size.
      - ``2048``
      - | Unit: megabyte (MB)
        | -1: No limit

    * - | ``GPU_MAX_WORKGROUP_SIZE``
        | Maximum number of workitems in a workgroup for GPU, 0 -use default
      - ``int``
      - 0

    * - | ``GPU_MIPMAP``
        | Enables GPU mipmap extension.
      - ``1``
      - | 0: Disable
        | 1: Enable

    * - | ``GPU_NUM_COMPUTE_RINGS``
        | GPU number of compute rings.
      - ``2``
      - | 0: Disable
        | 1, 2, ...: The number of compute rings.

    * - | ``GPU_NUM_MEM_DEPENDENCY``
        | Number of memory objects for dependency tracking.
      - ``256``
      -

    * - | ``GPU_PINNED_MIN_XFER_SIZE``
        | The minimal buffer size for pinned read/write transfers in MB.
      - ``128``
      - Unit: megabyte (MB)

    * - | ``GPU_PINNED_XFER_SIZE``
        | The buffer size for pinning in read/write transfers in MB.
      - ``32``
      - Unit: megabyte (MB)

    * - | ``GPU_PRINT_CHILD_KERNEL``
        | Prints the specified number of the child kernels.
      - ``0``
      -

    * - | ``GPU_RESOURCE_CACHE_SIZE``
        | The resource cache size in MB.
      - ``64``
      - Unit: megabyte (MB)

    * - | ``GPU_SINGLE_ALLOC_PERCENT``
        | Maximum size of a single allocation as percentage of total.
      - ``85``
      - 

    * - | ``GPU_STAGING_BUFFER_SIZE``
        | Size of the GPU staging buffer in megabyte (MB).
      - ``4``
      - Unit: megabyte (MB)

    * - | ``GPU_STREAMOPS_CP_WAIT``
        | Force the stream memory operation to wait on CP.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``GPU_USE_DEVICE_QUEUE``
        | Use a dedicated device queue for the actual submissions.
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
      - Unit: kilobyte (kB)
        
    * - | ``HIP_FORCE_DEV_KERNARG``
        | Force device memory for kernel args.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``HIP_HIDDEN_FREE_MEM``
        | Amount of memory to hide from the free memory reported by hipMemGetInfo.
      - ``0``
      - | 0: Disable
        | Unit: megabyte (MB)

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
        | Set initial heap size for device malloc.
      - ``8388608``
      - | Unit: Byte 
        | The default value corresponds to 8 megabyte (MB). 

    * - | ``HIP_LAUNCH_BLOCKING``
        | Used for serialization on kernel execution.
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
        | Force this to use Runtime code object unbundler.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``HIP_VISIBLE_DEVICES``
        | Only devices whose index is present in the sequence are visible to HIP
      - Unset by default.
      - 0,1,2: Depending on the number of devices on the system.

    * - | ``HIP_VMEM_MANAGE_SUPPORT``
        | Virtual Memory Management Support.
      - ``1``
      - | 0: Disable
        | 1: Enable

    * - | ``HIPCC_VERBOSE``
        | How much extra info to show during build. E.g: compiler flags, paths.
      - ``0``
      - 

    * - | ``HIPRTC_COMPILE_OPTIONS_APPEND``
        | Set compile options needed for hiprtc compilation.
      - Unset by default.
      - 

    * - | ``HIPRTC_LINK_OPTIONS_APPEND``
        | Set link options needed for hiprtc compilation.
      - Unset by default.
      - 

    * - | ``HIPRTC_USE_RUNTIME_UNBUNDLER``
        | Set this to ``true`` to force runtime unbundler in hiprtc.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``HSA_KERNARG_POOL_SIZE``
        | Kernel arguments pool size.
      - ``1048576``
      - | Unit: Byte
        | The default value corresponds to 1 megabyte (MB).

    * - | ``HSA_LOCAL_MEMORY_ENABLE``
        | Enable HSA device local memory usage.
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
        | Force memory resources to become resident at allocation time.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``PAL_EMBED_KERNEL_MD``
        | Enables writing kernel metadata into command buffers.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``PAL_FORCE_ASIC_REVISION``
        | Force a specific ASIC revision for all devices.
      - ``0``
      -

    * - | ``PAL_HIP_IPC_FLAG``
        | Enable interprocess flag for device allocation in PAL HIP
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``PAL_PREPINNED_MEMORY_SIZE``
        | Size of prepinned memory.
      - ``64``
      - | Unit: kilobyte (kB)

    * - | ``PAL_RGP_DISP_COUNT``
        | The number of dispatches for RGP capture with SQTT.
      - ``10000``
      -

    * - | ``REMOTE_ALLOC``
        | Use remote memory for the global heap allocation.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``ROC_ACTIVE_WAIT_TIMEOUT``
        | Forces active wait of GPU interrupt for the timeout.
      - ``0``
      - Unit: microseconds (us)

    * - | ``ROC_AQL_QUEUE_SIZE``
        | AQL queue size in AQL packets.
      - ``16384``
      - | Unit: Byte 
        | The default value corresponds to 16 kilobyte (kB). 

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
        | Sets a global CU mask, entered as hex value for all queues, Each active bit represents using one CU, e.g. ``0xf`` enables only 4 CUs
      - Unset by default.
      - 

    * - | ``ROC_HMM_FLAGS``
        | ROCm HMM configuration flags.
      - ``0``
      - 

    * - | ``ROC_P2P_SDMA_SIZE``
        | The minimum size in KB for P2P transfer with SDMA.
      - ``1024``
      - | Unit: kilobyte (kB) 
        | The default value corresponds to 1 megabyte (MB).

    * - | ``ROC_SIGNAL_POOL_SIZE``
        | Initial size of HSA signal pool
      - ``uint``
      - 32

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
        | Use fine grain kernel args segment for supported ASICs.
      - ``1``
      - | 0: Disable
        | 1: Enable

    * - | ``ROCPROFILER_REGISTER_ROOT``
        | The path to the rocProfiler.
      - Unset by default.
      - 

AMD rocclr environment variables at debug build:

.. list-table::
    :header-rows: 1
    :widths: 35,14,51

    * - **Environment variable**
      - **Default value**
      - **Usage**

    * - | ``AMD_OCL_SUBST_OBJFILE``
        | Specify binary substitution config file for OpenCL.
      - Unset by default.
      - 

    * - | ``CPU_MEMORY_ALIGNMENT_SIZE``
        | Size in bytes for the default alignment for guarded memory on CPU.
      - 256
      - Unit: Byte 

    * - | ``CPU_MEMORY_GUARD_PAGE_SIZE``
        | Size in KB of CPU memory guard page.
      - ``64``
      - | Unit: kilobyte (kB)
        | The default value corresponds to 64 kilobyte (kB). 

    * - | ``CPU_MEMORY_GUARD_PAGES``
        | Use guard pages for CPU memory
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``MEMOBJ_BASE_ADDR_ALIGN``
        | Alignment of the base address of any allocate memory object.
      - ``4096``
      - | Unit: Byte 
        | The default value corresponds to 4 kilobyte (kB). 

    * - | ``PARAMETERS_MIN_ALIGNMENT``
        | Minimum alignment required for the abstract parameters stack.
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
      - Unset by default.
      - ``0,GPU-DEADBEEFDEADBEEF``

    * - | ``HSA_SCRATCH_MEM``
        | Maximum amount of scratch memory that can be used per process per gpu.
      -
      -

    * - | ``HSA_XNACK``
        | Turning on XNACK by setting the environment variable HSA_XNACK=1
      - Unset by default.
      - ``1``

    * - | ``HSA_CU_MASK``
        | Sets the mask on a lower level of queue creation in the driver, 
        | this mask will also be set for queues being profiled.
      - Unset by default.
      - ``1:0-8``

rocPRIM Environment Variables
=============================

Environment variables of rocPRIM library.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Usage**

    * - | ``HIP_DIR``
        | The path of the HIP SDK on Microsoft Windows, if ``HIP_PATH``
      - ``C:/hip``

    * - | ``HIP_PATH``
        | The path of the HIP SDK on Microsoft Windows. On linux the ``ROCM_PATH``
        | environment variable used to set different ROCm root path.
      - ``C:/hip``

    * - | ``VCPKG_PATH``
        | The path of the vcpkg package manager on Microsoft Windows. On linux 
        | this environment variable has no effect.
      - ``C:/github/vcpkg``

    * - | ``ROCM_PATH``
        | The path of the installed ROCm software stack on linux. On Microsoft 
        | Windows the ``HIP_DIR`` environment variable used to set 
        | different HIP SDK path.
      - ``/opt/rocm``

    * - | ``ROCM_CMAKE_PATH``
        | The path of the installed ROCm cmake file on Microsoft Windows.
      - ``C:/hipSDK``

    * - | ``HIPCC_COMPILE_FLAGS_APPEND``
        | Extra amdclang++ compiler flags on linux. Ignored, if CXX environment
        | variable is set.
      - By default it's empty.

    * - | ``ROCPRIM_USE_HMM``
        | The tests suite uses unified memory, if it's set to 1 during the tests
        | run.
      - By default it's empty.

    * - | ``CTEST_RESOURCE_GROUP_0``
        | Used by CI, and helps to group the tests for different CI steps. Most
        | users should ignore this.
      - By default it's empty.

hipCUB Environment Variables
============================

Environment variables of hipCUB library.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Usage**

    * - | ``HIP_DIR``
        | The path of the HIP SDK on Microsoft Windows.
      - ``C:/hip``

    * - | ``HIP_PATH``
        | The path of the HIP SDK on Microsoft Windows. On linux the ``ROCM_PATH``
        | environment variable used to set different ROCm root path.
      - ``C:/hip``

    * - | ``VCPKG_PATH``
        | The path of the vcpkg package manager on Microsoft Windows. On linux 
        | this environment variable has no effect.
      - ``C:/github/vcpkg``

    * - | ``ROCM_PATH``
        | The path of the installed ROCm software stack on linux. On Microsoft 
        | Windows the ``HIP_DIR`` environment variable used to set 
        | different HIP SDK path.
      - ``/opt/rocm``

    * - | ``HIPCC_COMPILE_FLAGS_APPEND``
        | Extra amdclang or amdclang++ compiler flags on linux. 
        | amdclang++ ignores this, if CXX environment variable is set.
        | amdclang ignores this, if CC environment variable is set.
      - Unset by default.

    * - | ``HIPCUB_USE_HMM``
        | The tests suite uses unified memory, if it's set to 1 during the tests
        | run.
      - Unset by default.

    * - | ``CTEST_RESOURCE_GROUP_0``
        | Used by CI, and helps to group the tests for different CI steps. Most
        | users should ignore this.
      - Unset by default.

rocThrust Environment Variables
===============================

Environment variables of rocThrust library.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Usage**

    * - | ``HIP_DIR``
        | The path of the HIP SDK on Microsoft Windows.
      - ``C:/hip``

    * - | ``HIP_PATH``
        | The path of the HIP SDK on Microsoft Windows. On linux the ``ROCM_PATH``
        | environment variable used to set different ROCm root path.
      - ``C:/hip``

    * - | ``VCPKG_PATH``
        | The path of the vcpkg package manager on Microsoft Windows. On linux 
        | this environment variable has no effect.
      - ``C:/github/vcpkg``

    * - | ``ROCM_PATH``
        | The path of the installed ROCm software stack on linux. On Microsoft 
        | Windows the ``HIP_DIR`` environment variable used to set 
        | different HIP SDK path.
      - ``/opt/rocm``

    * - | ``ROCTHRUST_USE_HMM``
        | The tests unified memory allocation usage
      - default: ``C:/hipSDK``

    * - | ``CTEST_RESOURCE_GROUP_0``
        | The path of the installed ROCm cmake file on windows
      - default: ``C:/hipSDK``

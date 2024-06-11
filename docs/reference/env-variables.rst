.. meta::
    :description: Environment variables reference
    :keywords: AMD, ROCm, environment variables, environment, reference

.. role:: cpp(code)
   :language: cpp

.. _env-variables-reference:

*************************************************************
Environment variables reference
*************************************************************

ROCm common environment variables
=================================

Environment variables that impact libraries over the ROCm software stack. 

.. list-table::
    :header-rows: 1

    * - Environment variable
      - Values

    * - | ``HIP_DIR``
        | The path of the HIP package. Specifically the location of hipConfig.cmake
        | or hip-config.cmake.
      -

    * - | ``HIP_PATH``
        | The path of the HIP SDK.
      -

    * - | ``HIP_OFFICIAL_BUILD``
        | Enable/Disable for mainline/staging builds. End users should not set this.
      -

    * - | ``HIP_PLATFORM``
        | The platform HIP is targeting. If ``HIP_PLATFORM`` is not set, then hipcc
        | will attempt to auto-detect based on if nvcc is found.
      - ``amd``, ``nvidia``

    * - | ``ROCM_BUILD_ID``
        | Build ID to identify the release of a given package. End users should
        | not set this.
      -

    * - | ``ROCM_PATH``
        | The path of the installed ROCm software stack.
      - default: ``/opt/rocm``

clr environment variables
=========================

Environment variables affecting all backends of project clr.

.. list-table::
    :widths: 85,15
    :header-rows: 1

    * - Environment variable
      - Values

    * - | ``HIP_PLATFORM``
        | The platform HIP is targeting. If ``HIP_PLATFORM`` is not set, then hipcc will attempt to auto-detect based on if nvcc is found.
      - ``amd``, ``nvidia``

    * - | ``HSA_DISABLE_CACHE``
        | Used to disable L2 cache.
      - ON, OFF

    * - | ``ROCM_HEADER_WRAPPER_WERROR``
        | Causes errors to be emitted instead of warnings.
      - ON, OFF

Environment variables affecting the opencl backend of project clr.

.. list-table::
    :widths: 85,15
    :header-rows: 1

    * - Environment variable
      - Values

    * - | ``CPACK_DEBIAN_PACKAGE_RELEASE``
        | This is the numbering of the Debian package itself, i.e. the version of the packaging and not the version of the content.
      - 1, 2, 3...

    * - | ``CPACK_RPM_PACKAGE_RELEASE``
        | This is the numbering of the RPM package itself, i.e. the version of the packaging and not the version of the content.
      - 1, 2, 3...

Environment variables affecting the hipamd backend of project clr.

.. list-table::
    :widths: 85,15
    :header-rows: 1

    * - Environment variable
      - Values

    * - | ``HIP_FORCE_QUEUE_PROFILING``
        | Used to run the app as if it were run in rocprof. Forces command queue profiling on by default.
      - ON, OFF

    * - | ``HSA_OVERRIDE_GFX_VERSION``
        | Override the target version. Used to enable HIP usage on unsupported hardware.
      - 11.0.0, 10.3.0

    * - | ``HSAKMT_DEBUG_LEVEL``
        | When set to the highest level, the system will print memory allocation info.
      - 1, 2, ... 7

    * - | ``ROCM_LIBPATCH_VERSION``
        | The ROCm version in the format of an integer. The format is
        | :cpp:`MAJOR * 10000 + MINOR * 100 + PATCH`
      - 50000, 60020...

    * - | ``ROCM_RPATH``
        | The rpath for ROCm libraries.
      - 

rocclr environment variables
----------------------------

AMD rocclr environment variables at release build:

.. list-table::
    :widths: 70,15,15
    :header-rows: 1

    * - Environment variable
      - Variable type
      - Default value

    * - | ``AMD_CPU_AFFINITY``
        | Reset CPU affinity of any runtime threads
      - ``bool``
      - ``false``

    * - | ``AMD_DIRECT_DISPATCH``
        | Enable direct kernel dispatch.
      - ``bool``
      - ``false``

    * - | ``AMD_GPU_FORCE_SINGLE_FP_DENORM``
        | Force denorm for single precision: -1 - don't force, 0 - disable, 1 - enable
      - ``int``
      - -1

    * - | ``AMD_LOG_LEVEL``
        | The default log level
      - ``int``
      - 0

    * - | ``AMD_LOG_LEVEL_FILE``
        | Set output file for AMD_LOG_LEVEL. Default is stderr
      - ``cstring``
      - 

    * - | ``AMD_LOG_MASK``
        | The mask to enable specific kinds of logs
      - ``uint``
      - ``0X7fffffff``

    * - | ``AMD_OCL_BUILD_OPTIONS``
        | Set the options for clBuildProgram and clCompileProgram, override
      - ``cstring``
      - 0

    * - | ``AMD_OCL_BUILD_OPTIONS_APPEND``
        | Append the options for clBuildProgram and clCompileProgram
      - ``cstring``
      - 0

    * - | ``AMD_OCL_LINK_OPTIONS``
        | Set the options for clLinkProgram, override
      - ``cstring``
      - 0

    * - | ``AMD_OCL_LINK_OPTIONS_APPEND``
        | Append the options for clLinkProgram
      - ``cstring``
      - 0

    * - | ``AMD_OCL_WAIT_COMMAND``
        | 1 = Enable a wait for every submitted command
      - ``bool``
      - ``false``

    * - | ``OCL_SET_SVM_SIZE``
        | set SVM space size for discrete GPU
      - ``uint``
      - 4*16384

    * - | ``OCL_STUB_PROGRAMS``
        | 1 = Enables OCL programs stubing
      - ``bool``
      - ``false``

    * - | ``OPENCL_VERSION``
        | Force GPU opencl version
      - ``uint``
      - 200

    * - | ``AMD_OPT_FLUSH``
        | Kernel flush option, ``0x0`` = Use system-scope fence operations. ``0x1`` = Use device-scope fence operations when possible.
      - ``uint``
      - 1

    * - | ``AMD_SERIALIZE_COPY``
        | Serialize copies, ``0x1`` = Wait for completion before enqueue, ``0x2`` = Wait for completion after enqueue ``0x3`` = both
      - ``uint``
      - 0

    * - | ``AMD_SERIALIZE_KERNEL``
        | Serialize kernel enqueue, ``0x1`` = Wait for completion before enqueue, ``0x2`` = Wait for completion after enqueue ``0x3`` = both
      - ``uint``
      - 0

    * - | ``AMD_THREAD_TRACE_ENABLE``
        | Enable thread trace extension
      - ``bool``
      - ``true``

    * - | ``CL_KHR_FP64``
        | Enable/Disable support for double precision
      - ``bool``
      - ``true``

    * - | ``CQ_THREAD_STACK_SIZE``
        | The default command queue thread stack size
      - ``size_t``
      - 256*Ki

    * - | ``CUDA_VISIBLE_DEVICES``
        | Only devices whose index is present in the sequence are visible to CUDA
      - ``cstring``
      - 

    * - | ``DEBUG_CLR_GRAPH_PACKET_CAPTURE``
        | Enable/Disable graph packet capturing
      - ``bool``
      - ``false``

    * - | ``DEBUG_CLR_LIMIT_BLIT_WG``
        | Limit the number of workgroups in blit operations
      - ``uint``
      - 16

    * - | ``DISABLE_DEFERRED_ALLOC``
        | Disables deferred memory allocation on device
      - ``bool``
      - ``false``

    * - | ``GPU_ADD_HBCC_SIZE``
        | Add HBCC size to the reported device memory
      - ``bool``
      - ``false``

    * - | ``GPU_ANALYZE_HANG``
        | 1 = Enables GPU hang analysis
      - ``bool``
      - ``false``

    * - | ``GPU_BLIT_ENGINE_TYPE``
        | Blit engine type: 0 - Default, 1 - Host, 2 - CAL, 3 - Kernel
      - ``uint``
      - ``0x0``

    * - | ``GPU_CP_DMA_COPY_SIZE``
        | Set maximum size of CP DMA copy in KiB
      - ``uint``
      - 1

    * - | ``GPU_DEBUG_ENABLE``
        | Enables collection of extra info for debugger at some performance cost
      - ``bool``
      - ``false``

    * - | ``GPU_DEVICE_ORDINAL``
        | Select the device ordinal, a comma separated list of available devices
      - ``cstring``
      - 

    * - | ``GPU_DUMP_BLIT_KERNELS``
        | Dump the kernels for blit manager
      - ``bool``
      - ``false``

    * - | ``GPU_DUMP_CODE_OBJECT``
        | Enable dump code object
      - ``bool``
      - ``false``

    * - | ``GPU_ENABLE_COOP_GROUPS``
        | Enables cooperative group launch
      - ``bool``
      - ``true``

    * - | ``GPU_ENABLE_HW_P2P``
        | Enables HW P2P path
      - ``bool``
      - ``false``

    * - | ``GPU_ENABLE_LC``
        | Enables LC path
      - ``bool``
      - ``true``

    * - | ``GPU_ENABLE_PAL``
        | Enables PAL backend. 0 - ROC, 1 - PAL, 2 - ROC or PAL
      - ``uint``
      - 2

    * - | ``GPU_ENABLE_WAVE32_MODE``
        | Enables Wave32 compilation in HW if available
      - ``bool``
      - ``true``

    * - | ``GPU_ENABLE_WGP_MODE``
        | Enables WGP Mode in HW if available
      - ``bool``
      - ``true``

    * - | ``GPU_FORCE_BLIT_COPY_SIZE``
        | Size in KB of the threshold below which to force blit instead for sdma
      - ``size_t``
      - 0

    * - | ``GPU_FORCE_QUEUE_PROFILING``
        | Force command queue profiling by default
      - ``bool``
      - ``false``

    * - | ``GPU_FLUSH_ON_EXECUTION``
        | Submit commands to HW on every operation. 0 - Disable, 1 - Enable
      - ``bool``
      - ``false``

    * - | ``GPU_IMAGE_BUFFER_WAR``
        | Enables image buffer workaround
      - ``bool``
      - ``true``

    * - | ``GPU_IMAGE_DMA``
        | Enable DRM DMA for image transfers
      - ``bool``
      - ``true``

    * - | ``GPU_MAX_COMMAND_BUFFERS``
        | The maximum number of command buffers allocated per queue
      - ``uint``
      - 8

    * - | ``GPU_MAX_HEAP_SIZE``
        | Set maximum size of the GPU heap to % of board memory
      - ``uint``
      - 100

    * - | ``GPU_MAX_HW_QUEUES``
        | The maximum number of HW queues allocated per device
      - ``uint``
      - 4

    * - | ``GPU_MAX_REMOTE_MEM_SIZE``
        | Maximum size , in Ki that allows device memory substitution with system
      - ``uint``
      - 2

    * - | ``GPU_MAX_SUBALLOC_SIZE``
        | The maximum size accepted for suballocaitons in KB
      - ``size_t``
      - 4096

    * - | ``GPU_MAX_USWC_ALLOC_SIZE``
        | Set a limit in Mb on the maximum USWC allocation size, -1 = No limit
      - ``uint``
      - 2048

    * - | ``GPU_MAX_WORKGROUP_SIZE``
        | Maximum number of workitems in a workgroup for GPU, 0 -use default
      - ``int``
      - 0

    * - | ``GPU_MIPMAP``
        | Enables GPU mipmap extension
      - ``bool``
      - ``true``

    * - | ``GPU_NUM_COMPUTE_RINGS``
        | GPU number of compute rings. 0 - disabled, 1, 2, ... - the number of compute rings
      - ``uint``
      - 2

    * - | ``GPU_NUM_MEM_DEPENDENCY``
        | Number of memory objects for dependency tracking
      - ``size_t``
      - 256

    * - | ``GPU_PINNED_MIN_XFER_SIZE``
        | The minimal buffer size for pinned read/write transfers in MiB
      - ``size_t``
      - 128

    * - | ``GPU_PINNED_XFER_SIZE``
        | The buffer size for pinning in read/write transfers in MiB
      - ``size_t``
      - 32

    * - | ``GPU_PRINT_CHILD_KERNEL``
        | Prints the specified number of the child kernels
      - ``uint``
      - 0

    * - | ``GPU_RESOURCE_CACHE_SIZE``
        | The resource cache size in MB
      - ``size_t``
      - 64

    * - | ``GPU_SINGLE_ALLOC_PERCENT``
        | Maximum size of a single allocation as percentage of total  
      - ``uint``
      - 85

    * - | ``GPU_STAGING_BUFFER_SIZE``
        | Size of the GPU staging buffer in MiB
      - ``uint``
      - 4

    * - | ``GPU_STREAMOPS_CP_WAIT``
        | Force the stream wait memory operation to wait on CP.
      - ``bool``
      - ``false``

    * - | ``GPU_USE_DEVICE_QUEUE``
        | Use a dedicated device queue for the actual submissions
      - ``bool``
      - ``false``

    * - | ``GPU_WAVES_PER_SIMD``
        | Force the number of waves per SIMD , 1-10
      - ``uint``
      - 0

    * - | ``GPU_XFER_BUFFER_SIZE``
        | Transfer buffer size for image copy optimization in KB
      - ``size_t``
      - 0

    * - | ``HIP_FORCE_DEV_KERNARG``
        | Force device mem for kernel args.
      - ``bool``
      - 0

    * - | ``HIP_HIDDEN_FREE_MEM``
        | Reserve free mem reporting in Mb, 0 = Disable
      - ``uint``
      - 0

    * - | ``HIP_HOST_COHERENT``
        | Coherent memory in ``hipHostMalloc``
      - ``uint``
      - 0

    * - | ``HIP_INITIAL_DM_SIZE``
        | Set initial heap size for device malloc. The default value corresponds to 8 MiB
      - ``size_t``
      - 8388608

    * - | ``HIP_LAUNCH_BLOCKING``
        | Serialize kernel enqueue :cpp:`0x1` = Wait for completion after enqueue, same as :cpp:`AMD_SERIALIZE_KERNEL=2`
      - ``uint``
      - 0

    * - | ``HIP_MEM_POOL_SUPPORT``
        | Enables memory pool support in HIP
      - ``bool``
      - ``false``

    * - | ``HIP_MEM_POOL_USE_VM``
        | Enables memory pool support in HIP
      - ``bool``
      - | ``true`` on Windows, 
        | ``false`` on other OS

    * - | ``HIP_USE_RUNTIME_UNBUNDLER``
        | Force this to use Runtime code object unbundler.
      - ``bool``
      - ``false``

    * - | ``HIP_VISIBLE_DEVICES``
        | Only devices whose index is present in the sequence are visible to HIP
      - ``cstring``
      - 

    * - | ``HIP_VMEM_MANAGE_SUPPORT``
        | Virtual Memory Management Support
      - ``bool``
      - ``true``

    * - | ``HIPCC_VERBOSE``
        | How much extra info to show during build. E.g: compiler flags, paths.
      - ``uint``
      - 0

    * - | ``HIPRTC_COMPILE_OPTIONS_APPEND``
        | Set compile options needed for hiprtc compilation
      - ``cstring``
      - 

    * - | ``HIPRTC_LINK_OPTIONS_APPEND``
        | Set link options needed for hiprtc compilation
      - ``cstring``
      - 

    * - | ``HIPRTC_USE_RUNTIME_UNBUNDLER``
        | Set this to ``true`` to force runtime unbundler in hiprtc.
      - ``bool``
      - ``false``

    * - | ``HSA_KERNARG_POOL_SIZE``
        | Kernarg pool size
      - ``uint``
      - 1024 * 1024

    * - | ``HSA_LOCAL_MEMORY_ENABLE``
        | Enable HSA device local memory usage
      - ``bool``
      - ``true``

    * - | ``PAL_DISABLE_SDMA``
        | 1 = Disable SDMA for PAL
      - ``bool``
      - ``false``

    * - | ``PAL_MALL_POLICY``
        | Controls the behaviour of allocations with respect to the MALL, 0 = MALL policy is decided by KMD, 1 = Allocations are never put through the MALL, 2 = Allocations will always be put through the MALL
      - ``uint``
      - 0

    * - | ``PAL_ALWAYS_RESIDENT``
        | Force memory resources to become resident at allocation time
      - ``bool``
      - ``false``

    * - | ``PAL_EMBED_KERNEL_MD``
        | Enables writing kernel metadata into command buffers.
      - ``bool``
      - ``false``

    * - | ``PAL_FORCE_ASIC_REVISION``
        | Force a specific ASIC revision for all devices
      - ``uint``
      - 0

    * - | ``PAL_HIP_IPC_FLAG``
        | Enable interprocess flag for device allocation in PAL HIP
      - ``bool``
      - ``false``

    * - | ``PAL_PREPINNED_MEMORY_SIZE``
        | Size in KBytes of prepinned memory
      - ``size_t``
      - 64

    * - | ``PAL_RGP_DISP_COUNT``
        | The number of dispatches for RGP capture with SQTT
      - ``uint``
      - 10000

    * - | ``REMOTE_ALLOC``
        | Use remote memory for the global heap allocation
      - ``bool``
      - ``false``

    * - | ``ROC_ACTIVE_WAIT_TIMEOUT``
        | Forces active wait of GPU interrupt for the timeout, us unit
      - ``uint``
      - 0

    * - | ``ROC_AQL_QUEUE_SIZE``
        | AQL queue size in AQL packets
      - ``uint``
      - 16384

    * - | ``ROC_CPU_WAIT_FOR_SIGNAL``
        | Enable CPU wait for dependent HSA signals.
      - ``bool``
      - ``true``

    * - | ``ROC_ENABLE_LARGE_BAR``
        | Enable Large Bar if supported by the device
      - ``bool``
      - ``true``

    * - | ``ROC_GLOBAL_CU_MASK``
        | Sets a global CU mask, entered as hex value for all queues, Each active bit represents using one CU, e.g. ``0xf`` enables only 4 CUs
      - ``cstring``
      - 

    * - | ``ROC_HMM_FLAGS``
        | ROCm HMM configuration flags
      - ``uint``
      - 0

    * - | ``ROC_P2P_SDMA_SIZE``
        | The minimum size in KB for P2P transfer with SDMA
      - ``uint``
      - 1024

    * - | ``ROC_SIGNAL_POOL_SIZE``
        | Initial size of HSA signal pool
      - ``uint``
      - 32

    * - | ``ROC_SKIP_KERNEL_ARG_COPY``
        | If ``true``, then runtime can skip kernel arg copy
      - ``bool``
      - ``false``

    * - | ``ROC_SYSTEM_SCOPE_SIGNAL``
        | Enable system scope for signals, uses interrupts.
      - ``bool``
      - ``true``

    * - | ``ROC_USE_FGS_KERNARG``
        | Use fine grain kernel args segment for supported ASICs
      - ``bool``
      - ``true``

    * - | ``ROCPROFILER_REGISTER_ROOT``
        | The path to the rocProfiler.
      - ``cstring``
      - 

AMD rocclr environment variables at debug build:

.. list-table::
    :widths: 65,15,20
    :header-rows: 1

    * - Environment variable
      - Variable type
      - Default value

    * - | ``AMD_OCL_SUBST_OBJFILE``
        | Specify binary substitution config file for OpenCL
      - ``cstring``
      - 0

    * - | ``CPU_MEMORY_ALIGNMENT_SIZE``
        | Size in bytes for the default alignment for guarded memory on CPU
      - ``size_t``
      - 256

    * - | ``CPU_MEMORY_GUARD_PAGE_SIZE``
        | Size in KB of CPU memory guard page
      - ``size_t``
      - 64

    * - | ``CPU_MEMORY_GUARD_PAGES``
        | Use guard pages for CPU memory
      - ``bool``
      - ``false``

    * - | ``MEMOBJ_BASE_ADDR_ALIGN``
        | Alignment of the base address of any allocate memory object. The default value corresponds to 4 KiB.
      - ``size_t``
      - 4096

    * - | ``PARAMETERS_MIN_ALIGNMENT``
        | Minimum alignment required for the abstract parameters stack
      - ``size_t``
      - 64 at ``__AVX512F__``, 32 at ``__AVX__`` and 16 at other cases

ROCR-Runtime environment variables
==================================

.. https://github.com/ROCm/ROCR-Runtime/blob/master/src/core/util/flag.h

AMD ROCR-Runtime environment variables:

.. list-table::
    :widths: 65,15,20
    :header-rows: 1

    * - Environment variable
      - Variable type
      - Default value

    * - ``HSA_CHECK_FLAT_SCRATCH``
      - 0, 1
      -

    * - ``HSA_ENABLE_VM_FAULT_MESSAGE``
      - 0, 1
      -

    * - ``HSA_ENABLE_QUEUE_FAULT_MESSAGE``
      - 0, 1
      -

    * - ``HSA_ENABLE_INTERRUPT``
      -
      -

    * - ``HSA_ENABLE_SDMA``
      -
      -

    * - ``HSA_ENABLE_PEER_SDMA``
      -
      -

    * - ``HSA_ENABLE_SDMA_GANG``
      -
      -

    * - ``HSA_ENABLE_SDMA_COPY_SIZE_OVERRIDE``
      -
      -

    * - ``ROCR_VISIBLE_DEVICES``
      -
      -

    * - ``HSA_RUNNING_UNDER_VALGRIND``
      -
      -

    * - ``HSA_SDMA_WAIT_IDLE``
      -
      -

    * - ``HSA_MAX_QUEUES``
      -
      -

    * - ``HSA_SCRATCH_MEM``
      -
      -

    * - ``HSA_SCRATCH_SINGLE_LIMIT``
      -
      -

    * - ``HSA_SCRATCH_SINGLE_LIMIT_ASYNC``
      -
      -

    * - ``HSA_ENABLE_SCRATCH_ASYNC_RECLAIM``
      -
      -

    * - ``HSA_ENABLE_SCRATCH_ALT``
      -
      -

    * - ``HSA_TOOLS_LIB``
      -
      -

    * - ``HSA_TOOLS_REPORT_LOAD_FAILURE``
      -
      -

    * - ``HSA_TOOLS_DISABLE_REGISTER``
      -
      -

    * - ``HSA_TOOLS_REPORT_REGISTER_FAILURE``
      -
      -

    * - ``HSA_DISABLE_FRAGMENT_ALLOCATOR``
      -
      -

    * - ``HSA_ENABLE_SDMA_HDP_FLUSH``
      -
      -

    * - ``HSA_REV_COPY_DIR``
      -
      -

    * - ``HSA_FORCE_FINE_GRAIN_PCIE``
      -
      -

    * - ``HSA_NO_SCRATCH_RECLAIM``
      -
      -

    * - ``HSA_NO_SCRATCH_THREAD_LIMITER``
      -
      -

    * - ``HSA_DISABLE_IMAGE``
      -
      -

    * - ``HSA_LOADER_ENABLE_MMAP_URI``
      -
      -

    * - ``HSA_FORCE_SDMA_SIZE``
      -
      -

    * - ``HSA_IGNORE_SRAMECC_MISREPORT``
      -
      -

    * - ``HSA_XNACK``
      -
      -

    * - ``HSA_ENABLE_DEBUG``
      -
      -

    * - ``HSA_CU_MASK_SKIP_INIT``
      -
      -

    * - ``HSA_COOP_CU_COUNT``
      -
      -

    * - ``HSA_DISCOVER_COPY_AGENTS``
      -
      -

    * - ``HSA_SVM_PROFILE``
      -
      -

    * - ``HSA_ENABLE_SRAMECC``
      -
      -

    * - ``HSA_IMAGE_PRINT_SRD``
      -
      -

    * - ``HSA_ENABLE_MWAITX``
      -
      -

    * - ``HSA_ENABLE_IPC_MODE_LEGACY``
      -
      -

    * - ``HSA_OVERRIDE_CPU_AFFINITY_DEBUG``
      -
      -

    * - ``HSA_CU_MASK``
      -
      -

rocPRIM environment variables
=============================

Environment variables of rocPRIM library.

.. list-table::
    :widths: 65,35
    :header-rows: 1

    * - Environment variable
      - Values

    * - | ``HIP_DIR``
        | The path of the HIP package. Specifically the location of hipConfig.cmake or hip-config.cmake.
      -

    * - | ``HIP_PATH``
        | The path of the HIP SDK.
      - | default: ``/opt/rocm`` on Linux, 
        | ``C:/hip`` on windows

    * - | ``ROCM_PATH``
        | The path of the installed ROCm software stack on linux
      - default: ``/opt/rocm``

    * - | ``ROCM_CMAKE_PATH``
        | The path of the installed ROCm cmake file on windows
      - default: ``C:/hipSDK``

    * - | ``HIPCC_COMPILE_FLAGS_APPEND``
        | The path of the installed ROCm cmake file on windows
      - default: ``C:/hipSDK``

    * - | ``ROCPRIM_USE_HMM``
        | The tests unified memory allocation usage
      - default: ``C:/hipSDK``

    * - | ``CTEST_RESOURCE_GROUP_0``
        | The path of the installed ROCm cmake file on windows
      - default: ``C:/hipSDK``

hipCUB environment variables
============================

Environment variables of hipCUB library.

.. list-table::
    :widths: 65,35
    :header-rows: 1

    * - Environment variable
      - Values

    * - | ``HIP_DIR``
        | The path of the HIP package. Specifically the location of hipConfig.cmake or hip-config.cmake.
      -

    * - | ``HIP_PATH``
        | The path of the HIP SDK.
      - | default: ``/opt/rocm`` on Linux, 
        | ``C:/hip`` on windows

    * - | ``ROCM_PATH``
        | The path of the installed ROCm software stack on linux
      - default: ``/opt/rocm``

    * - | ``ROCM_CMAKE_PATH``
        | The path of the installed ROCm cmake file on windows
      - default: ``C:/hipSDK``

    * - | ``HIPCC_COMPILE_FLAGS_APPEND``
        | The path of the installed ROCm cmake file on windows
      - default: ``C:/hipSDK``

    * - | ``ROCPRIM_USE_HMM``
        | The tests unified memory allocation usage
      - default: ``C:/hipSDK``

    * - | ``CTEST_RESOURCE_GROUP_0``
        | The path of the installed ROCm cmake file on windows
      - default: ``C:/hipSDK``

rocThrust environment variables
===============================

Environment variables of rocThrust library.

.. list-table::
    :widths: 65,35
    :header-rows: 1

    * - Environment variable
      - Values

    * - | ``HIP_DIR``
        | The path of the HIP package. Specifically the location of hipConfig.cmake or hip-config.cmake.
      -

    * - | ``HIP_PATH``
        | The path of the HIP SDK.
      - | default: ``/opt/rocm`` on Linux, 
        | ``C:/hip`` on windows

    * - | ``ROCM_PATH``
        | The path of the installed ROCm software stack on linux
      - default: ``/opt/rocm``

    * - | ``ROCM_CMAKE_PATH``
        | The path of the installed ROCm cmake file on windows
      - default: ``C:/hipSDK``

    * - | ``HIPCC_COMPILE_FLAGS_APPEND``
        | The path of the installed ROCm cmake file on windows
      - default: ``C:/hipSDK``

    * - | ``ROCPRIM_USE_HMM``
        | The tests unified memory allocation usage
      - default: ``C:/hipSDK``

    * - | ``CTEST_RESOURCE_GROUP_0``
        | The path of the installed ROCm cmake file on windows
      - default: ``C:/hipSDK``

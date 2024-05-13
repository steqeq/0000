.. meta::
    :description: Environment variables reference
    :keywords: AMD, ROCm, environment variables, environment, reference

.. _env-variables-reference:

*************************************************************
Environment variables reference
*************************************************************

General overview
==========================================

Environment variables that impact libraries in ROCm software stack

.. list-table:: Environment variables that impact libraries in ROCm software stack
    :header-rows: 1
    :name: clr-env-variables-general-table

    *
      - Environment variable
      - Values
      - Description
    *
      - HIP_DIR
      - 
      - The path of the HIP package. Specifically the location of hipConfig.cmake or hip-config.cmake.
    *
      - HIP_OFFICIAL_BUILD
      - 
      - Enable/Disable for mainline/staging builds. End users should not set this.
    *
      - HIP_PATH
      - 
      - The path of the HIP SDK.
    *
      - HIP_PLATFORM
      - ``amd``, ``nvidia``
      - The platform HIP is targeting. If HIP_PLATFORM is not set, then hipcc will attempt to auto-detect based on if nvcc is found.
    *
      - ROCM_BUILD_ID
      - 
      - Build ID to identify the release of a given package. End users should not set this.
    *
      - ROCM_PATH
      - /opt/rocm
      - The path of the installed ROCm software stack.

ROCm
==========================================

Environment variables in ROCm library.


clr
==========================================

Environment variables affecting all backends of project clr.

.. list-table:: Environment variables affecting all backends of project clr
    :header-rows: 1
    :name: clr-env-variables-all-table

    *
      - Environment variable
      - Values
      - Description
    *
      - HIP_PLATFORM
      - ``amd``, ``nvidia``
      - The platform HIP is targeting. If HIP_PLATFORM is not set, then hipcc will attempt to auto-detect based on if nvcc is found.
    *
      - HSA_DISABLE_CACHE
      - ON, OFF
      - Used to disable L2 cache.
    *
      - ROCM_HEADER_WRAPPER_WERROR
      - ON, OFF
      - Causes errors to be emitted instead of warnings.

Environment variables affecting the opencl backend of project clr.

.. list-table:: Environment variables affecting the opencl backend of project clr
    :header-rows: 1
    :name: clr-env-variables-opencl-table

    *
      - Environment variable
      - Values
      - Description
    *
      - CPACK_DEBIAN_PACKAGE_RELEASE
      - 1, 2, 3...
      - This is the numbering of the Debian package itself, i.e. the version of the packaging and not the version of the content.
    *
      - CPACK_RPM_PACKAGE_RELEASE
      - 1, 2, 3...
      - This is the numbering of the RPM package itself, i.e. the version of the packaging and not the version of the content.


Environment variables affecting the hipamd backend of project clr.

.. list-table:: Environment variables affecting the hipamd backend of project clr
    :header-rows: 1
    :name: clr-env-variables-hipamd-table

    *
      - Environment variable
      - Values
      - Description
    *
      - HIP_FORCE_QUEUE_PROFILING
      - ON, OFF
      - Used to run the app as if it were run in rocprof. Forces command queue profiling on by default.
    *
      - HSA_OVERRIDE_GFX_VERSION
      - 11.0.0, 10.3.0
      - Override the target version. Used to enable HIP usage on unsupported hardware.
    *
      - HSAKMT_DEBUG_LEVEL
      - 1, 2, ... 7
      - When set to the highest level, the system will print memory allocation info.
    *
      - ROCM_LIBPATCH_VERSION
      - 50000, 60020...
      - The ROCm version in the format of an integer. The format is MAJOR * 10000 + MINOR * 100 + PATCH
    *
      - ROCM_RPATH
      - 
      - The rpath for ROCm libraries.

rocclr environment variables
----------------------------------

AMD rocclr environment variables at release build:

.. list-table:: AMD rocclr environment variables
    :header-rows: 1
    :name: rocclr-env-variables-release-table

    *
      - Environment variable
      - Variable type
      - Default value
      - Description
    *
      - AMD_CPU_AFFINITY
      - bool
      - FALSE
      - Reset CPU affinity of any runtime threads
    *
      - AMD_DIRECT_DISPATCH
      - bool
      - FALSE
      - Enable direct kernel dispatch.
    *
      - AMD_GPU_FORCE_SINGLE_FP_DENORM
      - int
      - -1
      - Force denorm for single precision: -1 - don't force, 0 - disable, 1 - enable
    *
      - AMD_LOG_LEVEL
      - int
      - 0
      - The default log level
    *
      - AMD_LOG_LEVEL_FILE
      - cstring
      - 
      - Set output file for AMD_LOG_LEVEL. Default is stderr
    *
      - AMD_LOG_MASK
      - uint
      - 0X7FFFFFFF
      - The mask to enable specific kinds of logs
    *
      - AMD_OCL_BUILD_OPTIONS
      - cstring
      - 0
      - Set the options for clBuildProgram and clCompileProgram, override
    *
      - AMD_OCL_BUILD_OPTIONS_APPEND
      - cstring
      - 0
      - Append the options for clBuildProgram and clCompileProgram
    *
      - AMD_OCL_LINK_OPTIONS
      - cstring
      - 0
      - Set the options for clLinkProgram, override
    *
      - AMD_OCL_LINK_OPTIONS_APPEND
      - cstring
      - 0
      - Append the options for clLinkProgram
    *
      - AMD_OCL_WAIT_COMMAND
      - bool
      - FALSE
      - 1 = Enable a wait for every submitted command
    *
      - AMD_OPT_FLUSH
      - uint
      - 1
      - Kernel flush option, 0x0 = Use system-scope fence operations. 0x1 = Use device-scope fence operations when possible.
    *
      - AMD_SERIALIZE_COPY
      - uint
      - 0
      - Serialize copies, 0x1 = Wait for completion before enqueue, 0x2 = Wait for completion after enqueue 0x3 = both
    *
      - AMD_SERIALIZE_KERNEL
      - uint
      - 0
      - Serialize kernel enqueue, 0x1 = Wait for completion before enqueue, 0x2 = Wait for completion after enqueue 0x3 = both
    *
      - AMD_THREAD_TRACE_ENABLE
      - bool
      - TRUE
      - Enable thread trace extension
    *
      - CL_KHR_FP64
      - bool
      - TRUE
      - Enable/Disable support for double precision
    *
      - CQ_THREAD_STACK_SIZE
      - size_t
      - 256*Ki
      - The default command queue thread stack size
    *
      - CUDA_VISIBLE_DEVICES
      - cstring
      - 
      - Only devices whose index is present in the sequence are visible to CUDA
    *
      - DEBUG_CLR_GRAPH_PACKET_CAPTURE
      - bool
      - FALSE
      - Enable/Disable graph packet capturing
    *
      - DEBUG_CLR_LIMIT_BLIT_WG
      - uint
      - 16
      - Limit the number of workgroups in blit operations
    *
      - DISABLE_DEFERRED_ALLOC
      - bool
      - FALSE
      - Disables deferred memory allocation on device
    *
      - GPU_ADD_HBCC_SIZE
      - bool
      - FALSE
      - Add HBCC size to the reported device memory
    *
      - GPU_ANALYZE_HANG
      - bool
      - FALSE
      - 1 = Enables GPU hang analysis
    *
      - GPU_BLIT_ENGINE_TYPE
      - uint
      - 0x0
      - Blit engine type: 0 - Default, 1 - Host, 2 - CAL, 3 - Kernel
    *
      - GPU_CP_DMA_COPY_SIZE
      - uint
      - 1
      - Set maximum size of CP DMA copy in KiB
    *
      - GPU_DEBUG_ENABLE
      - bool
      - FALSE
      - Enables collection of extra info for debugger at some performance cost
    *
      - GPU_DEVICE_ORDINAL
      - cstring
      - 
      - Select the device ordinal, a comma separated list of available devices
    *
      - GPU_DUMP_BLIT_KERNELS
      - bool
      - FALSE
      - Dump the kernels for blit manager
    *
      - GPU_DUMP_CODE_OBJECT
      - bool
      - FALSE
      - Enable dump code object
    *
      - GPU_ENABLE_COOP_GROUPS
      - bool
      - TRUE
      - Enables cooperative group launch
    *
      - GPU_ENABLE_HW_P2P
      - bool
      - FALSE
      - Enables HW P2P path
    *
      - GPU_ENABLE_LC
      - bool
      - TRUE
      - Enables LC path
    *
      - GPU_ENABLE_PAL
      - uint
      - 2
      - Enables PAL backend. 0 - ROC, 1 - PAL, 2 - ROC or PAL
    *
      - GPU_ENABLE_WAVE32_MODE
      - bool
      - TRUE
      - Enables Wave32 compilation in HW if available
    *
      - GPU_ENABLE_WGP_MODE
      - bool
      - TRUE
      - Enables WGP Mode in HW if available
    *
      - GPU_FORCE_BLIT_COPY_SIZE
      - size_t
      - 0
      - Size in KB of the threshold below which to force blit instead for sdma
    *
      - GPU_FORCE_QUEUE_PROFILING
      - bool
      - FALSE
      - Force command queue profiling by default
    *
      - GPU_FLUSH_ON_EXECUTION
      - bool
      - FALSE
      - Submit commands to HW on every operation. 0 - Disable, 1 - Enable
    *
      - GPU_IMAGE_BUFFER_WAR
      - bool
      - TRUE
      - Enables image buffer workaround
    *
      - GPU_IMAGE_DMA
      - bool
      - TRUE
      - Enable DRM DMA for image transfers
    *
      - GPU_MAX_COMMAND_BUFFERS
      - uint
      - 8
      - The maximum number of command buffers allocated per queue
    *
      - GPU_MAX_HEAP_SIZE
      - uint
      - 100
      - Set maximum size of the GPU heap to % of board memory
    *
      - GPU_MAX_HW_QUEUES
      - uint
      - 4
      - The maximum number of HW queues allocated per device"
    *
      - GPU_MAX_REMOTE_MEM_SIZE
      - uint
      - 2
      - Maximum size , in Ki that allows device memory substitution with system
    *
      - GPU_MAX_SUBALLOC_SIZE
      - size_t
      - 4096
      - The maximum size accepted for suballocaitons in KB
    *
      - GPU_MAX_USWC_ALLOC_SIZE
      - uint
      - 2048
      - Set a limit in Mb on the maximum USWC allocation size, -1 = No limit
    *
      - GPU_MAX_WORKGROUP_SIZE
      - int
      - 0
      - Maximum number of workitems in a workgroup for GPU, 0 -use default
    *
      - GPU_MIPMAP
      - bool
      - TRUE
      - Enables GPU mipmap extension
    *
      - GPU_NUM_COMPUTE_RINGS
      - uint
      - 2
      - GPU number of compute rings. 0 - disabled, 1, 2, ... - the number of compute rings
    *
      - GPU_NUM_MEM_DEPENDENCY
      - size_t
      - 256
      - Number of memory objects for dependency tracking
    *
      - GPU_PINNED_MIN_XFER_SIZE
      - size_t
      - 128
      - The minimal buffer size for pinned read/write transfers in MiB
    *
      - GPU_PINNED_XFER_SIZE
      - size_t
      - 32
      - The buffer size for pinning in read/write transfers in MiB
    *
      - GPU_PRINT_CHILD_KERNEL
      - uint
      - 0
      - Prints the specified number of the child kernels
    *
      - GPU_RESOURCE_CACHE_SIZE
      - size_t
      - 64
      - The resource cache size in MB
    *
      - GPU_SINGLE_ALLOC_PERCENT
      - uint
      - 85
      - Maximum size of a single allocation as percentage of total
    *
      - GPU_STAGING_BUFFER_SIZE
      - uint
      - 4
      - Size of the GPU staging buffer in MiB
    *
      - GPU_STREAMOPS_CP_WAIT
      - bool
      - FALSE
      - Force the stream wait memory operation to wait on CP.
    *
      - GPU_USE_DEVICE_QUEUE
      - bool
      - FALSE
      - Use a dedicated device queue for the actual submissions
    *
      - GPU_WAVES_PER_SIMD
      - uint
      - 0
      - Force the number of waves per SIMD , 1-10
    *
      - GPU_XFER_BUFFER_SIZE
      - size_t
      - 0
      - Transfer buffer size for image copy optimization in KB
    *
      - HIP_FORCE_DEV_KERNARG
      - bool
      - 0
      - Force device mem for kernel args.
    *
      - HIP_HIDDEN_FREE_MEM
      - uint
      - 0
      - Reserve free mem reporting in Mb, 0 = Disable
    *
      - HIP_HOST_COHERENT
      - uint
      - 0
      - Coherent memory in hipHostMalloc
    *
      - HIP_INITIAL_DM_SIZE
      - size_t
      - 8388608
      - Set initial heap size for device malloc. The default value corresponds to 8 MiB
    *
      - HIP_LAUNCH_BLOCKING
      - uint
      - 0
      - Serialize kernel enqueue 0x1 = Wait for completion after enqueue, same as AMD_SERIALIZE_KERNEL=2
    *
      - HIP_MEM_POOL_SUPPORT
      - bool
      - FALSE
      - Enables memory pool support in HIP
    *
      - HIP_MEM_POOL_USE_VM
      - bool
      - IS_WINDOWS
      - Enables memory pool support in HIP
    *
      - HIP_USE_RUNTIME_UNBUNDLER
      - bool
      - FALSE
      - Force this to use Runtime code object unbundler.
    *
      - HIP_VISIBLE_DEVICES
      - cstring
      - 
      - Only devices whose index is present in the sequence are visible to HIP
    *
      - HIP_VMEM_MANAGE_SUPPORT
      - bool
      - TRUE
      - Virtual Memory Management Support
    *
      - HIPCC_VERBOSE
      - uint
      - 0
      - How much extra info to show during build. E.g: compiler flags, paths.
    *
      - HIPRTC_COMPILE_OPTIONS_APPEND
      - cstring
      - 
      - Set compile options needed for hiprtc compilation
    *
      - HIPRTC_LINK_OPTIONS_APPEND
      - cstring
      - 
      - Set link options needed for hiprtc compilation
    *
      - HIPRTC_USE_RUNTIME_UNBUNDLER
      - bool
      - FALSE
      - Set this to true to force runtime unbundler in hiprtc.
    *
      - HSA_KERNARG_POOL_SIZE
      - uint
      - 1024 * 1024
      - Kernarg pool size
    *
      - HSA_LOCAL_MEMORY_ENABLE
      - bool
      - TRUE
      - Enable HSA device local memory usage
    *
      - OCL_SET_SVM_SIZE
      - uint
      - 4*16384
      - set SVM space size for discrete GPU
    *
      - OCL_STUB_PROGRAMS
      - bool
      - FALSE
      - 1 = Enables OCL programs stubing
    *
      - OPENCL_VERSION
      - uint
      - 200
      - Force GPU opencl version
    *
      - PAL_DISABLE_SDMA
      - bool
      - FALSE
      - 1 = Disable SDMA for PAL
    *
      - PAL_MALL_POLICY
      - uint
      - 0
      - Controls the behaviour of allocations with respect to the MALL, 0 = MALL policy is decided by KMD, 1 = Allocations are never put through the MALL, 2 = Allocations will always be put through the MALL
    *
      - PAL_ALWAYS_RESIDENT
      - bool
      - FALSE
      - Force memory resources to become resident at allocation time
    *
      - PAL_EMBED_KERNEL_MD
      - bool
      - FALSE
      - Enables writing kernel metadata into command buffers.
    *
      - PAL_FORCE_ASIC_REVISION
      - uint
      - 0
      - Force a specific ASIC revision for all devices
    *
      - PAL_HIP_IPC_FLAG
      - bool
      - FALSE
      - Enable interprocess flag for device allocation in PAL HIP
    *
      - PAL_PREPINNED_MEMORY_SIZE
      - size_t
      - 64
      - Size in KBytes of prepinned memory
    *
      - PAL_RGP_DISP_COUNT
      - uint
      - 10000
      - The number of dispatches for RGP capture with SQTT
    *
      - REMOTE_ALLOC
      - bool
      - FALSE
      - Use remote memory for the global heap allocation
    *
      - ROC_ACTIVE_WAIT_TIMEOUT
      - uint
      - 0
      - Forces active wait of GPU interrupt for the timeout, us unit
    *
      - ROC_AQL_QUEUE_SIZE
      - uint
      - 16384
      - AQL queue size in AQL packets
    *
      - ROC_CPU_WAIT_FOR_SIGNAL
      - bool
      - TRUE
      - Enable CPU wait for dependent HSA signals.
    *
      - ROC_ENABLE_LARGE_BAR
      - bool
      - TRUE
      - Enable Large Bar if supported by the device
    *
      - ROC_GLOBAL_CU_MASK
      - cstring
      - 
      - Sets a global CU mask, entered as hex value for all queues, Each active bit represents using one CU, e.g. 0xf enables only 4 CUs
    *
      - ROC_HMM_FLAGS
      - uint
      - 0
      - ROCm HMM configuration flags
    *
      - ROC_P2P_SDMA_SIZE
      - uint
      - 1024
      - The minimum size in KB for P2P transfer with SDMA
    *
      - ROC_SIGNAL_POOL_SIZE
      - uint
      - 32
      - Initial size of HSA signal pool
    *
      - ROC_SKIP_KERNEL_ARG_COPY
      - bool
      - FALSE
      - If true, then runtime can skip kernel arg copy
    *
      - ROC_SYSTEM_SCOPE_SIGNAL
      - bool
      - TRUE
      - Enable system scope for signals, uses interrupts.
    *
      - ROC_USE_FGS_KERNARG
      - bool
      - TRUE
      - Use fine grain kernel args segment for supported ASICs
    *
      - ROCPROFILER_REGISTER_ROOT
      - cstring
      - 
      - The path to the rocProfiler.

AMD rocclr environment variables at debug build:

.. list-table:: AMD rocclr environment variables
    :header-rows: 1
    :name: rocclr-env-variables-debug-table

    *
      - Environment variable
      - Variable type
      - Default value
      - Description
    *
      - AMD_OCL_SUBST_OBJFILE
      - cstring
      - 0
      - Specify binary substitution config file for OpenCL
    *
      - CPU_MEMORY_ALIGNMENT_SIZE
      - size_t
      - 256
      - Size in bytes for the default alignment for guarded memory on CPU
    *
      - CPU_MEMORY_GUARD_PAGE_SIZE
      - size_t
      - 64
      - Size in KB of CPU memory guard page
    *
      - CPU_MEMORY_GUARD_PAGES
      - bool
      - FALSE
      - Use guard pages for CPU memory
    *
      - MEMOBJ_BASE_ADDR_ALIGN
      - size_t
      - 4096
      - Alignment of the base address of any allocate memory object. The default value corresponds to 4 KiB.
    *
      - PARAMETERS_MIN_ALIGNMENT
      - size_t
      - NATIVE_ALIGNMENT_SIZE
      - Minimum alignment required for the abstract parameters stack

.. meta::
   :description: How to fine-tune LLMs with ROCm
   :keywords: ROCm, LLM, fine-tuning, usage, tutorial, Triton, kernel, performance, optimization

*************************
Optimizing Triton kernels
*************************

This section introduces the general steps for `Triton <https://openai.com/index/triton/>`_ kernel optimization. Broadly,
Triton kernel optimization is similar to HIP and CUDA kernel optimization.

.. _fine-tuning-llms-triton-hardware-resource-utilization:

Hardware resource utilization
=============================

Each accelerator or GPU has multiple Compute Units (CUs) and various CUs do computation in parallel. So, how many CUs
can a compute kernel can allocate its task to? For AMD MI300X, the grid should have at least 1024 thread blocks or
workgroups.

To increase hardware utilization and maximize parallelism, it is necessary to design algorithms that can exploit more
parallelism. One approach to achieving this is by using larger split-K techniques for General Matrix Multiply (GEMM)
operations, which can further distribute the computation across more CUs, thereby enhancing performance.

.. tip::

   Hardware resources can be queried with the command ``rocminfo`` (in the ``/opt/rocm/bin`` directory). For instance,
   query the number of CUs, # of SIMD, and wavefront size using the following commands.

   .. code-block:: shell

      rocminfo | grep "Compute Unit"

      rocminfo | grep "SIMD"

      rocminfo | grep "Wavefront Size"

   On an MI300X device, there are 304 CUs, 4 SIMD per CU, and the wavefront size (warp size) is 64. See :doc:`Hardware
   specifications <rocm:reference/gpu-arch-specs>` for a full list of AMD accelerators and GPUs.

.. _fine-tuning-llms-triton-kernel-configs-env-vars:

Auto-tunable kernel configurations and environment variables
===========================================================

This section relates to the amount of memory access and computation assigned to each CU. It is related to the usage of
LDS, registers and the scheduling of different tasks on a CU.

The following kernel arguments can be tuned.

``num_stages=n``
   On AMD accelerators, set ``num_stages`` according to the following rules:

   -  For kernels with single GEMM, set to 0.

   -  For kernels with two GEMMs fused (Flash Attention, or any other kernel
      that fuses 2 GEMMs), set to 1.

   -  For kernels that fuse a single GEMM with another non GEMM operator
      (for example ReLU activation), set to 0.

   -  For kernels that have no GEMMs, set to 1.

``waves_per_eu=n``
   See :ref:`Understand/Compute the occupancy of the
   kernel <fine-tuning-llms-occupancy-of-kernel>` for more
   information about how to compute occupancy. It hints to the compiler to
   reduce VGPR so that occupancy = n could be achieved. This only helps if
   both of the following are satisfied:

   -  The occupancy of the kernel is limited by VGPR usage.

   -  The current VGPR usage is only a few above a boundary in table 1.

   For example, according to the table, the available VGPR is 512 per
   Execution Unit (EU), and VGPU is allocated at the unit of 16. If the
   current VGPR usage is 170, the actual requested VGPR will be 176, so the
   occupancy is only 2 waves/CU since 176 x 3 > 512. Then if you set
   ``waves_per_eu`` to 3, the LLVM backend tries to bring VGPR usage down so
   that it might fit 3 waves/EU.

``BLOCK_M``, ``BLOCK_N``, ``BLOCK_K``
   Tile sizes need to be tuned. You want tile sizes large enough to
   maximize the efficiency of memory-to-computation ratio, but small enough
   to parallelize the greatest number of workgroups at the grid level.

``matrix_instr_nonkdim``
   This is an experimental feature for FA-like kernels. It can choose the
   size of MFMA instruction used. For GEMM kernels on AMD MI300X,
   ``mfma_16x16`` performs better than ``mfma_32x32``, even for large tile/GEMM
   sizes.

   -  ``Matrix_instr_nonkdim = 16``: ``mfma_16x16`` is used

   -  ``Matrix_instr_nonkdim = 32``: ``mfma_32x32`` is used

``OPTIMIZE_EPILOGUE``
   This is an environment variable that should be turned on (set to 1) in
   most cases. It removes the ``convert_layout`` in the epilogue. By default,
   the results of MFMA instruction are converted to blocked layout, which
   leads to ``global_store`` with maximum vector length, that is
   ``global_store_dwordx4``.

   This is done implicitly with LDS as the intermediate buffer to achieve
   data exchange between threads. Padding is used in LDS to avoid bank
   conflicts. This usually leads to extra LDS usage, which might reduce
   occupancy. Setting ``OPTIMIZE_EPILOGUE=1`` will have the effect of storing
   the result in the MFMA layout. This reduces the efficiency of global
   stores but has an insignificant influence on kernel execution time.

   .. note::

      This variable is not turned on by default because it only
      works with ``tt.store`` but not ``tt.atomic_add``, which is used in split-k and
      stream-k GEMM kernels. In the future, it might be enabled with
      ``tt.atomic_add`` and turned on by default.

.. _fine-tuning-llms-triton-memory-access-efficiency:

Memory access efficiency
========================

The GPU contains global memory, local data share (LDS), and registers. Global memory has high access latency, but is
large. LDS access has much lower latency, but is smaller. Register access is the fastest yet smallest among the three.

So, the data in global memory should be loaded and stored as few times as possible. If different threads in a block
need to access the same data, these data should be first transferred from global memory to LDS, then accessed by
different threads in a workgroup.

.. _fine-tuning-llms-triton-ir-analysis:

IR analysis
===========

In Triton, there are several layouts including blocked, shared, sliced, and MFMA.

From the Triton GPU IR (intermediate representation), you can know in which memory each computation is
performed. The following is a snippet of IR from the Flash Attention decode ``int4`` key-value program. It is to de-quantize
the ``int4`` key-value from the ``int4`` data type to ``fp16``.

.. code-block::

   %190 = tt.load %189 {cache = 1 : i32, evict = 1 : i32, isVolatile =
   false} : tensor<1x64xi32, #blocked6> loc(#loc159)

   %266 = arith.andi %190, %cst_28 : tensor<1x64xi32, #blocked6>
   loc(#loc250)

   %267 = arith.trunci %266 : tensor<1x64xi32, #blocked6> to
   tensor<1x64xi16, #blocked6> loc(#loc251)

   %268 = tt.bitcast %267 : tensor<1x64xi16, #blocked6> -> tensor<1x64xf16,
   #blocked6> loc(#loc252)

   %269 = triton_gpu.convert_layout %268 : (tensor<1x64xf16, #blocked6>) ->
   tensor<1x64xf16, #shared1> loc(#loc252)

   %270 = tt.trans %269 : (tensor<1x64xf16, #shared1>) -> tensor<64x1xf16,
   #shared2> loc(#loc194)

   %276 = triton_gpu.convert_layout %270 : (tensor<64x1xf16, #shared2>) ->
   tensor<64x1xf16, #blocked5> loc(#loc254)

   %293 = arith.mulf %276, %cst_30 : tensor<64x1xf16, #blocked5>
   loc(#loc254)

   %295 = arith.mulf %292, %294 : tensor<64x32xf16, #blocked5> loc(#loc264)

   %297 = arith.addf %295, %296 : tensor<64x32xf16, #blocked5> loc(#loc255)

   %298 = triton_gpu.convert_layout %297 : (tensor<64x32xf16, #blocked5>)
   -> tensor<64x32xf16, #shared1> loc(#loc255)

   %299 = tt.trans %298 : (tensor<64x32xf16, #shared1>) ->
   tensor<32x64xf16, #shared2> loc(#loc196)

   %300 = triton_gpu.convert_layout %299 : (tensor<32x64xf16, #shared2>) ->
   tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth
   = 4}>> loc(#loc197)

From the IR, you can see ``i32`` data is loaded from global memory to
registers. With a few element-wise operations in registers, then it is
stored in shared memory for the transpose operation, which needs data
movement across different threads. With the transpose done, it is loaded
from LDS to register again, and with a few more element-wise operations,
they are stored in LDS again. The last step is to load from LDS to registers
and convert to the dot-operand layout.

From the IR, you can see that it uses the LDS twice: one for the
transpose, and the other to convert the blocked layout to a dot-operand layout.

Assembly analysis
=================

In the ISA, ensure ``global_load_dwordx4`` is used, especially when the
load happens in a loop.

In most cases, the LDS load and store should use ``_b128`` as well to
minimize the number of LDS access instructions. Note that upstream (or backend) might not have ``_b128`` LDS read/write,
so it uses ``_b64``. For most cases, no matter if you use fork or upstream,
the LDS access should have ``_b64`` vector width.

The AMD ISA has the ``s_waitcnt`` instruction to synchronize the dependency
of memory access and computations. The ``s_waitcnt`` instruction can
have two signals, typically in the context of Triton:

* ``lgkmcnt(n):`` `lgkm` stands for LDS, GDS, Constant and Message.

  In this context, it is often related to LDS access. The number ``n`` here means the number of such accesses that can
  be left out to continue. For example, 0 means all ``lgkm`` access must finish before continuing, and 1 means only 1
  ``lgkm`` access can be still running asynchronously before proceeding.

* ``vmcnt(n):`` `vm` means vector memory.

  This happens when vector memory is accessed, for example, when global load moves from global memory to vector memory.
  Again, the number ``n`` here means the number of accesses that can be left out to continue.

Generally recommended guidelines are as follows.

*  Vectorize memory access as much as possible.

*  Ensure synchronization is done efficiently.

*  Overlap of instructions to hide latency, but it requires thoughtful
   analysis of the algorithms.

*  If you find inefficiencies, you can trace it back to LLVM IR, TTGIR
   and even TTIR to see where the problem comes from. If you find it
   during compiler optimization, activate the MLIR dump and check which
   optimization pass caused the problem.

.. _fine-tuning-llms-occupancy-of-kernel:

Understand and compute the occupancy of the kernel
==================================================

1. Get the VGPR count, search for ``.vgpr_count`` in the ISA. For example, N.

2. Get the allocated LDS following the steps. For example, L for the kernel.

  a. ``export MLIR_ENABLE_DUMP=1``

  b. ``rm -rf ~/.triton/cache``

  c. ``python kernel.py | | grep "triton_gpu.shared = " | tail -n 1``

  d. You should see something like ``triton_gpu.shared = 65536``, indicating 65536 bytes of LDS are allocated for the
     kernel.

3. Get number of waves per workgroup using the following steps (say you got ``nW``)

  a. ``export MLIR_ENABLE_DUMP=1``

  b. ``rm -rf ~/.triton/cache``

  c. ``python kernel.py | | grep "triton_gpu.num-warps " | tail -n 1``

  d. You should see something like ``“triton_gpu.num-warps" = 8``, indicating 8 waves per workgroup.

4. Compute occupancy limited by VGPR based on N according to table 1 in this link. For example, waves per EU as
   ``occ_vgpr``.

5. Compute occupancy limited by LDS based on L by: ``occ_lds = floor(65536 / L)``.

6. Then the occupancy is ``occ = min(floor(occ_vgpr * 4 / nW), occ_lds) * nW / 4``

  a. ``occ_vgpr \* 4`` gives the total number of waves on all 4 execution units (SIMDs)
  per CU

  b. ``floor(occ_vgpr * 4 / nW)`` gives the occupancy of workgroups per CU
  regrading VGPR usage

  c. The true ``occ`` is the minimum of the two.

PyTorch ``inductor`` Triton tuning knobs
========================================

To enable a ``gemm/conv`` lowering to Triton, it requires use of ``inductor``’s ``max_autotune`` mode. This benchmarks a
static list of Triton configurations (``conv`` configurations for max auto-tune + ``matmul`` configurations for max auto-tune) and uses the
fastest for each shape. Note that the Triton is not used if regular :doc:`MIOpen <miopen:index>` or :doc:`rocBLAS
<rocblas:index>` is faster for a specific operation.

``torch._inductor.config.max_autotune = True`` or
``TORCHINDUCTOR_MAX_AUTOTUNE=1``

Or, for more fine-grained control:

``torch._inductor.config.max_autotune.pointwise = True`` - to enable tuning for ``pointwise``/``reduction`` ops

``torch._inductor.config.max_autotune_gemm = True`` - to enable tuning or lowering of ``mm``/``conv``s

``torch._inductor.max_autotune_gemm_backends/TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS``
- to select the candidate backends for mm auto-tuning Defaults to
``TRITON,ATEN``, NV also includes CUTLASS tuning option. Limiting this to
“TRITON” might improve performance by enabling more fused mm kernels
instead of going to rocBLAS

For ``mm tuning coordinate_descent`` tuning might improve performance,
which attempts

``torch._inductor.config.coordinate_descent_tuning = True`` or ``TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1``

Inference can see large improvements on AMD GPUs by utilizing
\`torch._inductor.config.freezing=True`/TORCHINDUCTOR_FREEZING=1, which
in-lines weights as constants and enables constant folding optimizations.

Enabling ``inductor``’s cpp_wrapper might improve overhead. This generates
C++ code which launches Triton binaries directly with
``hipModuleLaunchKernel`` and relies on `hipification`.

For NHWC convolutions workloads
``torch._inductor.config.layout_optimization=True`` or ``TORCHINDUCTOR_LAYOUT_OPTIMIZATION=``
can help be enforcing channels_last format throughout the graph avoiding
any additional transposes added by ``inductor``. Note that
``PYTORCH_MIOPEN_SUGGEST_NHWC=1`` is recommended if using this.

Extracting the Triton kernel ``TORCH_COMPILE_DEBUG`` creates a
``torch_compile_debug/`` directory at current path, in the ``output_code.py``
the code-strings for the Triton kernels that are defined. Manual work is
then required to strip out the kernel and create kernel
compilation and launch via Triton.

For advanced ``matmul`` or ``conv`` configuration tuning, the ``inductor-gemm-tuner`` can
help. This implements the Triton ``conv``/``mm`` implementations used upstream
and allows specification of inputs and configuration tuning search space if new
tunings are found that can be added to the auto-tune list.

Miscellaneous
=============

Performance-critical HIP provides an environment variable, ``export HIP_FORCE_DEV_KERNARG=1``,
that can put HIP kernel arguments directly to
device memory to reduce the latency of accessing kernel arguments. It
can reduce 2 to 3 μs for some kernels. Setting this variable for the FA
decode containing ``splitK`` and reduced kernels can reduce the total time
by around 6 μs in the benchmark test.

Set the clock to deterministic. Use the command ``rocm-smi --setperfdeterminism 1900`` to set the max clock speed to
1900MHz instead of the default 2100MHz. This can reduce the chance of clock speed decrease due to chip high temperature
by setting a lower cap. You can restore this setting to its default value with ``rocm-smi -r``.

Set `numa` auto-balance. Run the command ``cat /proc/sys/kernel/numa_balancing`` to check the current settings. An output
of ``0`` indicates this setting is available. If output is ``1``, run the command
``sudo sh -c \\'echo 0 > /proc/sys/kernel/numa_balancing`` to set this.

For these settings, we created a script to do ‘set’, ‘reset’, ‘checking’
of the above environments. The script is located at ``env_check.sh``.

.. code-block:: shell

   #!/bin/bash

   function print_usage {

   echo " Usage: env_set.sh set/reset/check"

   echo " set: configure the settings in this script"

   echo " reset: reset to default settings"

   echo " check: check the current settings"

   }

   function set_env {

   export HIP_FORCE_DEV_KERNARG=1

   rocm-smi --setperfdeterminism 1900

   sudo sh -c echo 0 > /proc/sys/kernel/numa_balancing

   }

   function reset_env {

   unset HIP_FORCE_DEV_KERNARG

   rocm-smi -r

   sudo sh -c echo 1 > /proc/sys/kernel/numa_balancing

   }

   function check_env {

   echo ""

   echo "---------------------------------------------------------------"

   echo ""

   # check the flag to force kernel to be on device memory

   echo "1. Check forcing kernel args on device memory"

   dev_kernarg=$(env \| grep HIP_FORCE_DEV_KERNARG)

   if [ -z $dev_kernarg ]

   then

   echo " no setting for forcing kernel args on device memory"

   echo " run the command \\"export HIP_FORCE_DEV_KERNARG=1\" to force it"

   else

   echo " env var \\"HIP_FORCE_DEV_KERNARG\" for forcing kernel args on
   device"

   echo " memory is set, we have HIP_FORCE_DEV_KERNARG="
   $HIP_FORCE_DEV_KERNARG

   if [ "$HIP_FORCE_DEV_KERNARG" -eq 0 ]

   then

   echo " env var HIP_FORCE_DEV_KERNARG is 0, set it to 1 by:"

   echo " command \\"export HIP_FORCE_DEV_KERNARG=1\""

   fi

   fi

   echo ""

   echo ""

   echo "2. Set perfdeterminism, highest frequency"

   echo " run the command \\"rocm-smi -a \| grep sclk\" to check highest
   frequency."

   echo " you can run the command \\"rocm-smi --setperfdeterminism # (for
   example 1900)\" to"

   echo " set clock frequency limit to get minimal performance, which is
   more reproducible"

   echo " you can restore the setting by running \\"rocm-smi
   --resetperfdeterminism\""

   echo ""

   echo ""

   echo "3. Check numa autobalance"

   autobal=$(cat /proc/sys/kernel/numa_balancing)

   if [ $autobal -ne 0 ]

   then

   echo " run the command \\"sudo sh -c \\'echo 0 >
   /proc/sys/kernel/numa_balancing\'\""

   echo " to set numa autobalance".

   echo " you can disable it with \\"sudo sh -c \\'echo 1 >
   /proc/sys/kernel/numa_balancing\'\""

   else

   echo " numa autobalance is checked with:"

   echo " (cat /proc/sys/kernel/numa_balancing)=0"

   fi

   echo ""

   echo "---------------------------------------------------------------"

   echo ""

   }

   if [ $# -eq 0 ]

   then

   echo " \\"env_set.sh -h\" for help info"

   print_usage

   exit 1

   fi

   input=$1

   if [ $1 == "set" ]

   then

   set_env

   elif [ $1 == "reset" ]

   then

   reset_env

   elif [ $1 == "check" ]

   then

   check_env

   else

   print_usage

   fi

TunableOp has been merged into PyTorch. The behavior of TunableOp is
easily manipulated through environment variables, though you could use
the C++ interface of ``at::cuda::tunable::getTuningContext()``. A Python
interface to the ``TuningContext`` does not yet exist.

The default is 0, which means only 1 iteration is attempted.

There’s an overhead to tuning. To try and minimize the overhead, only a
limited number of iterations of a given operation are attempted. If you
set this to 10, each solution for a given operation can run as many
iterations as possible within 10ms. There is a hard-coded upper limit of
100 iterations attempted per solution. This is a tuning parameter; if
you want the tunings to be chosen based on an average over multiple
iterations, increase the allowed tuning duration.


.. meta::
   :description: How to fine-tune LLMs with ROCm
   :keywords: ROCm, LLM, fine-tuning, usage, tutorial, Triton, kernel, performance, optimization

*************************
Optimizing Triton kernels
*************************

This section introduces the general steps for `Triton <https://openai.com/index/triton/>`_ kernel optimization. Broadly,
Triton kernel optimization is similar to HIP and CUDA kernel optimization.

.. _fine-tuning-llms-triton-memory-access-efficiency:

Memory access efficiency
========================

The accelerator or GPU contains global memory, local data share (LDS), and registers. Global memory has high access
latency, but is large. LDS access has much lower latency, but is smaller. Register access is the fastest yet smallest
among the three.

So, the data in global memory should be loaded and stored as few times as possible. If different threads in a block
need to access the same data, these data should be first transferred from global memory to LDS, then accessed by
different threads in a workgroup.

.. _fine-tuning-llms-triton-hardware-resource-utilization:

Hardware resource utilization
=============================

Each accelerator or GPU has multiple Compute Units (CUs) and various CUs do computation in parallel. So, how many CUs
can a compute kernel can allocate its task to? For the :doc:`AMD MI300X accelerator <../../reference/gpu-arch-specs>`, the
grid should have at least 1024 thread blocks or workgroups.

.. figure:: ../../data/how-to/fine-tuning-llms/compute-unit.png

   Schematic representation of a CU in the CDNA2 or CDNA3 architecture.

To increase hardware utilization and maximize parallelism, it is necessary to design algorithms that can exploit more
parallelism. One approach to achieving this is by using larger split-K techniques for General Matrix Multiply (GEMM)
operations, which can further distribute the computation across more CUs, thereby enhancing performance.

.. tip::

   You can query hardware resources with the command ``rocminfo`` (in the ``/opt/rocm/bin`` directory). For instance,
   query the number of CUs, number of SIMD, and wavefront size using the following commands.

   .. code-block:: shell

      rocminfo | grep "Compute Unit"

      rocminfo | grep "SIMD"

      rocminfo | grep "Wavefront Size"

   On an MI300X device, there are 304 CUs, 4 SIMD per CU, and the wavefront size (warp size) is 64. See :doc:`Hardware
   specifications <../../reference/gpu-arch-specs>` for a full list of AMD accelerators and GPUs.

.. _fine-tuning-llms-triton-ir-analysis:

IR analysis
===========

In Triton, there are several layouts including *blocked*, *shared*, *sliced*, and *MFMA*.

From the Triton GPU IR (intermediate representation), you can know in which memory each computation is
performed. The following is a snippet of IR from the Flash Attention decode ``int4`` key-value program. It is to
de-quantize the ``int4`` key-value from the ``int4`` data type to ``fp16``.

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

From the IR, you can see ``i32`` data is loaded from global memory to registers. With a few element-wise operations in
registers, then it is stored in shared memory for the transpose operation, which needs data movement across different
threads. With the transpose done, it is loaded from LDS to register again, and with a few more element-wise operations,
they are stored in LDS again. The last step is to load from LDS to registers and convert to the dot-operand layout.

From the IR, you can see that it uses the LDS twice: one for the transpose, and the other to convert the blocked layout
to a dot-operand layout.

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

.. _fine-tuning-llms-triton-kernel-occupancy:

Kernel occupancy
================

1. Get the VGPR count, search for ``.vgpr_count`` in the ISA. For example, N.

2. Get the allocated LDS following the steps. For example, L for the kernel.

   a. ``export MLIR_ENABLE_DUMP=1``

   b. ``rm -rf ~/.triton/cache``

   c. ``python kernel.py | | grep "triton_gpu.shared = " | tail -n 1``

   d. You should see something like ``triton_gpu.shared = 65536``, indicating 65536 bytes of LDS are allocated for the
      kernel.

3. Get number of waves per workgroup using the following steps (say you got ``nW``).

   a. ``export MLIR_ENABLE_DUMP=1``

   b. ``rm -rf ~/.triton/cache``

   c. ``python kernel.py | | grep "triton_gpu.num-warps " | tail -n 1``

   d. You should see something like ``“triton_gpu.num-warps" = 8``, indicating 8 waves per workgroup.

4. Compute occupancy limited by VGPR based on N according to the following table. For example, waves per EU as
   ``occ_vgpr``.

.. _fine-tuning-llms-occupancy-vgpr-table:

.. figure:: ../../data/how-to/fine-tuning-llms/occupancy-vgpr.png
   :alt: Occupancy related to VGPR usage in an Instinct MI300X accelerator.
   :align: center

5. Compute occupancy limited by LDS based on L by: ``occ_lds = floor(65536 / L)``.

6. Then the occupancy is ``occ = min(floor(occ_vgpr * 4 / nW), occ_lds) * nW / 4``

   a. ``occ_vgpr \* 4`` gives the total number of waves on all 4 execution units (SIMDs)
      per CU.

   b. ``floor(occ_vgpr * 4 / nW)`` gives the occupancy of workgroups per CU
      regrading VGPR usage.

   c. The true ``occ`` is the minimum of the two.

.. _fine-tuning-llms-triton-kernel-configs-env-vars:

Auto-tunable kernel configurations and environment variables
============================================================

This section relates to the amount of :ref:`memory access <fine-tuning-llms-triton-memory-access-efficiency>` and
computation assigned to each CU. It is related to the usage of LDS, registers and the scheduling of different tasks on
a CU.

The following is a list of kernel arguments used for tuning.

``num_stages=n``
   Adjusts the number of pipeline stages for different types of kernels. On AMD accelerators, set ``num_stages``
   according to the following rules:

   * For kernels with a single GEMM, set to ``0``.

   * For kernels with two GEMMs fused (Flash Attention, or any other kernel
     that fuses 2 GEMMs), set to ``1``.

   * For kernels that fuse a single GEMM with another non-GEMM operator
     (for example ReLU activation), set to ``0``.

   * For kernels that have no GEMMs, set to ``1``.

``waves_per_eu=n``
   Helps to manage Vector General Purpose Registers (VGPR) usage to achieve desired occupancy levels. This argument
   hints to the compiler to reduce VGPR to achieve ``n`` occupancy. See
   :ref:`Kernel occupancy <fine-tuning-llms-triton-kernel-occupancy>` for more information about how to compute
   occupancy. 

   This argument is useful if:

   * The occupancy of the kernel is limited by VGPR usage.

   * The current VGPR usage is only a few above a boundary in
     :ref:`Occupancy related to VGPR usage in an Instinct MI300X accelerator <fine-tuning-llms-occupancy-vgpr-table>`.

   For example, according to the table, the available VGPR is 512 per Execution Unit (EU), and VGPU is allocated at the
   unit of 16. If the current VGPR usage is 170, the actual requested VGPR will be 176, so the
   occupancy is only 2 waves per CU since :math:`176 \times 3 > 512`. So, if you set
   ``waves_per_eu`` to 3, the LLVM backend tries to bring VGPR usage down so
   that it might fit 3 waves per EU.

``BLOCK_M``, ``BLOCK_N``, ``BLOCK_K``
   Tile sizes to be tuned to balance the memory-to-computation ratio. You want tile sizes large enough to
   maximize the efficiency of memory-to-computation ratio, but small enough to parallelize the greatest number of
   workgroups at the grid level.

``matrix_instr_nonkdim``
   Experimental feature for Flash Attention-like kernels that determines the size of the Matrix Fused Multiply-Add
   (MFMA) instruction used.

   -  ``Matrix_instr_nonkdim = 16``: ``mfma_16x16`` is used.

   -  ``Matrix_instr_nonkdim = 32``: ``mfma_32x32`` is used.

   For GEMM kernels on an AMD MI300X accelerator, ``mfma_16x16`` typically outperforms ``mfma_32x32``, even for large
   tile/GEMM sizes.

The following is an environment variable used for tuning.

``OPTIMIZE_EPILOGUE``
   Setting this variable to ``1`` can improve performance by removing the ``convert_layout`` operation in the epilogue.
   It should be turned on (set to ``1``) in most cases. Setting ``OPTIMIZE_EPILOGUE=1`` stores the MFMA instruction
   results in the MFMA layout directly; this comes at the cost of reduced global store efficiency, but the impact on
   kernel execution time is usually minimal.

   By default (``0``), the results of MFMA instruction are converted to blocked layout, which leads to ``global_store``
   with maximum vector length, that is ``global_store_dwordx4``.

   This is done implicitly with LDS as the intermediate buffer to achieve
   data exchange between threads. Padding is used in LDS to avoid bank
   conflicts. This usually leads to extra LDS usage, which might reduce
   occupancy.

   .. note::

      This variable is not turned on by default because it only
      works with ``tt.store`` but not ``tt.atomic_add``, which is used in split-k and
      stream-k GEMM kernels. In the future, it might be enabled with
      ``tt.atomic_add`` and turned on by default.

   See :ref:`IR analysis <fine-tuning-llms-triton-ir-analysis>`.

PyTorch ``inductor`` Triton tuning knobs
========================================

The following are suggestions for optimizing matrix multiplication (GEMM) and convolution (``conv``) operations in PyTorch
using ``inductor``, a part of the PyTorch compilation framework. The goal is to leverage Triton to achieve better
performance.

To enable a ``gemm``/``conv`` lowering to Triton, it requires use of ``inductor``’s ``max_autotune`` mode. This benchmarks a
static list of Triton configurations (``conv`` configurations for max auto-tune + ``matmul`` configurations for max
auto-tune) and uses the fastest for each shape. Note that the Triton is not used if regular :doc:`MIOpen <miopen:index>`
or :doc:`rocBLAS <rocblas:index>` is faster for a specific operation.

* Set ``torch._inductor.config.max_autotune = True`` or ``TORCHINDUCTOR_MAX_AUTOTUNE=1``.

* Or, for more fine-grained control:

  ``torch._inductor.config.max_autotune.pointwise = True``
     To enable tuning for ``pointwise``/``reduction`` ops.

  ``torch._inductor.config.max_autotune_gemm = True``
     To enable tuning or lowering of ``mm``/``conv``\s.

  ``torch._inductor.max_autotune_gemm_backends/TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS``
     To select the candidate backends for ``mm`` auto-tuning. Defaults to
     ``TRITON,ATEN,NV``. This also includes the ``CUTLASS`` tuning option. Limiting this to
     ``TRITON`` might improve performance by enabling more fused ``mm`` kernels
     instead of going to rocBLAS.

* For ``mm`` tuning, tuning ``coordinate_descent`` might improve performance.

  ``torch._inductor.config.coordinate_descent_tuning = True`` or ``TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1``

* Inference can see large improvements on AMD GPUs by utilizing
  ``torch._inductor.config.freezing=True`` or the ``TORCHINDUCTOR_FREEZING=1`` variable, which
  in-lines weights as constants and enables constant folding optimizations.

* Enabling ``inductor``’s cpp_wrapper might improve overhead. This generates
  C++ code which launches Triton binaries directly with
  ``hipModuleLaunchKernel`` and relies on `hipification`.

* For NHWC convolutions workloads
  ``torch._inductor.config.layout_optimization=True`` or ``TORCHINDUCTOR_LAYOUT_OPTIMIZATION=``
  can help be enforcing channels_last format throughout the graph avoiding
  any additional transposes added by ``inductor``. Note that
  ``PYTORCH_MIOPEN_SUGGEST_NHWC=1`` is recommended if using this.

* Extracting the Triton kernel ``TORCH_COMPILE_DEBUG`` creates a
  ``torch_compile_debug/`` directory at current path, in the ``output_code.py``
  the code-strings for the Triton kernels that are defined. Manual work is
  then required to strip out the kernel and create kernel
  compilation and launch via Triton.

* For advanced ``matmul`` or ``conv`` configuration tuning, the ``inductor-gemm-tuner`` can
  help. This implements the Triton ``conv``/``mm`` implementations used upstream
  and allows specification of inputs and configuration tuning search space if new
  tunings are found that can be added to the auto-tune list.

Other guidelines
================

* Performance-critical HIP provides an environment variable, ``export HIP_FORCE_DEV_KERNARG=1``,
  that can put HIP kernel arguments directly to
  device memory to reduce the latency of accessing kernel arguments. It
  can reduce 2 to 3 μs for some kernels. Setting this variable for the FA
  decode containing ``splitK`` and reduced kernels can reduce the total time
  by around 6 μs in the benchmark test.

* Set the clock to deterministic. Use the command ``rocm-smi --setperfdeterminism 1900`` to set the max clock speed to
  1900MHz instead of the default 2100MHz. This can reduce the chance of clock speed decrease due to chip high temperature
  by setting a lower cap. You can restore this setting to its default value with ``rocm-smi -r``.

* Set Non-Uniform Memory Access (NUMA) auto-balance. Run the command ``cat /proc/sys/kernel/numa_balancing`` to check the
  current setting. An output of ``0`` indicates this setting is available. If output is ``1``, run the command
  ``sudo sh -c \\'echo 0 > /proc/sys/kernel/numa_balancing`` to set this.

For these settings, the ``env_check.sh`` script automates the setting, resetting, and checking of the such
environments. Find the script at `<https://github.com/ROCm/triton/blob/rocm_env/scripts/amd/env_check.sh>`__.

.. _fine-tuning-llms-triton-tunableop:

TunableOp
---------
`TunableOp <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/tunable/README.md>`_
is a feature used to define and optimize kernels that can have tunable parameters. This is useful in
optimizing the performance of custom kernels by exploring different parameter configurations to find the most efficient
setup. See more about PyTorch TunableOp :ref:`Model acceleration libraries <fine-tuning-llms-pytorch-tunableop>`.

You can easily manipulate the behavior TunableOp through environment variables, though you could use the C++ interface
``at::cuda::tunable::getTuningContext()``. A Python interface to the ``TuningContext`` does not yet exist.

The default value is ``0``, which means only 1 iteration is attempted. Remember: there’s an overhead to tuning. To try
and minimize the overhead, only a limited number of iterations of a given operation are attempted. If you set this to
``10``, each solution for a given operation can run as many iterations as possible within 10ms. There is a hard-coded
upper limit of 100 iterations attempted per solution. This is a tuning parameter; if you want the tunings to be chosen
based on an average over multiple iterations, increase the allowed tuning duration.

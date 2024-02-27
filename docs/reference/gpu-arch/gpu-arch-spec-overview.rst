.. meta::
   :description: AMD Instinct™ GPU architecture information
   :keywords: Instinct, CDNA, GPU, architecture, VRAM, Compute Units, Cache, Registers, LDS, Register File

GPU architecture hardware specifications
########################################

The following table provides an overview over the hardware specifications for the AMD Instinct accelerators.

.. list-table:: AMD Instinct architecture specification table
    :header-rows: 1
    :name: instinct-arch-spec-table

    *
      - Model
      - Architecture
      - LLVM target name
      - VRAM
      - Compute Units
      - Wavefront Size
      - LDS
      - L3 Cache
      - L2 Cache
      - L1 Vector Cache
      - L1 Scalar Cache
      - L1 Instruction Cache
      - VGPR File
      - SGPR File
    *
      - MI300X
      - CDNA3
      - gfx941 or gfx942
      - 192 GiB
      - 304
      - 64
      - 64 KiB
      - 256 MiB
      - 32 MiB
      - 32 KiB
      - 16 KiB per 2 CUs
      - 64 KiB per 2 CUs
      - 512 KiB
      - 12.5 KiB
    *
      - MI300A
      - CDNA3
      - gfx940 or gfx942
      - 128 GiB
      - 228
      - 64
      - 64 KiB
      - 256 MiB
      - 24 MiB
      - 32 KiB
      - 16 KiB per 2 CUs
      - 64 KiB per 2 CUs
      - 512 KiB
      - 12.5 KiB
    *
      - MI250X
      - CDNA2
      - gfx90a
      - 128 GiB
      - 220 (110 per GCD)
      - 64
      - 64 KiB
      -
      - 16 MiB (8 MiB per GCD)
      - 16 KiB
      - 16 KiB per 2 CUs
      - 32 KiB per 2 CUs
      - 512 KiB
      - 12.5 KiB
    *
      - MI250
      - CDNA2
      - gfx90a
      - 128 GiB
      - 208
      - 64
      - 64 KiB
      -
      - 16 MiB (8 MiB per GCD)
      - 16 KiB
      - 16 KiB per 2 CUs
      - 32 KiB per 2 CUs
      - 512 KiB
      - 12.5 KiB
    *
       - MI210
       - CDNA2
       - gfx90a
       - 64 GiB
       - 104
       - 64
       - 64 KiB
       -
       - 8 MiB
       - 16 KiB
       - 16 KiB per 2 CUs
       - 32 KiB per 2 CUs
       - 512 KiB
       - 12.5 KiB
    *
      - MI100
      - CDNA
      - gfx908
      - 32 GiB
      - 120
      - 64
      - 64 KiB
      -
      - 8 MiB
      - 16 KiB
      - 16 KiB per 3 CUs
      - 32 KiB per 3 CUs
      - 256 KiB VGPR and 256 KiB AccVGPR
      - 12.5 KiB
    *
      - MI60
      - GCN 5.1
      - gfx906
      - 32 GiB
      - 64
      - 64
      - 64 KiB
      -
      - 4 MiB
      - 16 KiB
      - 16 KiB per 3 CUs
      - 32 KiB per 3 CUs
      - 256 KiB
      - 12.5 KiB
    *
      - MI50 (32GB)
      - GCN 5.1
      - gfx906
      - 32 GiB
      - 60
      - 64
      - 64 KiB
      -
      - 4 MiB
      - 16 KiB
      - 16 KiB per 3 CUs
      - 32 KiB per 3 CUs
      - 256 KiB
      - 12.5 KiB
    *
      - MI50 (16GB)
      - GCN 5.1
      - gfx906
      - 16 GiB
      - 60
      - 64
      - 64 KiB
      -
      - 4 MiB
      - 16 KiB
      - 16 KiB per 3 CUs
      - 32 KiB per 3 CUs
      - 256 KiB
      - 12.5 KiB
    *
      - MI25
      - GCN 5.0
      - gfx900
      - 16 GiB
      - 64
      - 64
      - 64 KiB
      -
      - 4 MiB
      - 16 KiB
      - 16 KiB per 3 CUs
      - 32 KiB per 3 CUs
      - 256 KiB
      - 12.5 KiB
    *
      - MI8
      - GCN 3.0
      - gfx803
      - 4 GiB
      - 64
      - 64
      - 64 KiB
      -
      - 2 MiB
      - 16 KiB
      - 16 KiB per 4 CUs
      - 32 KiB per 4 CUs
      - 256 KiB
      - 12.5 KiB
    *
      - MI6
      - GCN 4.0
      - gfx803
      - 16 GiB
      - 36
      - 64
      - 64 KiB
      -
      - 2 MiB
      - 16 KiB
      - 16 KiB per 4 CUs
      - 32 KiB per 4 CUs
      - 256 KiB
      - 12.5 KiB

Glossary
########

For a more detailed explanation refer to the :ref:`specific documents and guides <gpu-arch-documentation>`.

LLVM target name
  Argument to pass to clang in `--offload-arch` to compile code for the given architecture.
VRAM
  Amount of memory available on the GPU.
Compute Units
  Number of compute units on the GPU.
Wavefront Size
  Amount of work-items that execute in parallel on a single compute unit. This is equivalent to the warp size in HIP.
LDS
  The Local Data Share (LDS) is a low-latency, high-bandwidth scratch pad memory. It is local to the compute units, shared by all work-items in a work group. In HIP this is the shared memory, which is shared by all threads in a block.
L3 Cache
  Size of the level 3 cache. Shared by all compute units on the same GPU. Caches vector and scalar data and instructions.
L2 Cache
  Size of the level 3 cache. Shared by all compute units on the same GCD. Caches vector and scalar data and instructions.
L1 Vector Cache
  Size of the level 1 vector data cache. Local to a compute unit. Caches vector data.
L1 Scalar Cache
  Size of the level 1 scalar data cache. Usually shared by several compute units. Caches scalar data.
L1 Instruction Cache
  Size of the level 1 instruction cache. Usually shared by several compute units.
VGPR File
  Size of the Vector General Purpose Register (VGPR) file. Holds data used in vector instructions.
  GPUs with matrix cores also have AccVGPRs, which are Accumulation General Purpose Vector Registers, specifically used in matrix instructions.
SGPR File
  Size of the Scalar General Purpose Register (SGPR) file. Holds data used in scalar instructions.
GCD
  Graphics Compute Die.

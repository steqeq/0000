.. meta::
   :description: AMD Instinct™ GPU architecture information
   :keywords: Instinct, CDNA, GPU, architecture, VRAM, Compute Units, Cache, Registers, LDS, Register File

**********************************************************************************
GPU architecture hardware specifications
**********************************************************************************

The following tables provide an overview of the hardware specifications for AMD Instinct™
accelerators, AMD Radeon™ and AMD Radeon™ Pro GPUs.

.. dropdown:: AMD Instinct GPUs

  .. list-table::
      :header-rows: 1
      :name: instinct-arch-spec-table

      *
        - Model
        - Architecture
        - LLVM target name
        - VRAM
        - Compute Units
        - Warp Size
        - LDS
        - L3 Cache
        - L2 Cache
        - L1 Vector Cache
        - L1 Scalar Cache
        - L1 Instruction Cache
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
        - GCN5.1
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
        - GCN5.1
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
        - GCN5.1
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
        - GCN5.0
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
        - GCN3.0
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
        - GCN4.0
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

.. dropdown:: AMD Radeon Pro GPUs

  .. list-table::
      :header-rows: 1
      :name: radeon-pro-arch-spec-table

      *
        - Model
        - Architecture
        - LLVM target name
        - VRAM
        - Compute Units
        - Warp Size
        - LDS
        - Infinity Cache
        - L2 Cache
        - Graphics L1 Cache
        - L0 Vector Cache
        - L0 Scalar Cache
        - L0 Instruction Cache
        - VGPR File
        - SGPR File
      *
        - Radeon PRO W7900
        - RDNA3
        - gfx1100
        - 48 GiB
        - 96
        - 32
        - 128 KiB
        - 96 MiB
        - 6 MiB
        - 256 KiB
        - 32 KiB
        - 16 KiB
        - 32 KiB
        - 384 KiB
        - 20 KiB
      *
        - Radeon PRO W7800
        - RDNA3
        - gfx1100
        - 32 GiB
        - 70
        - 32
        - 128 KiB
        - 64 MiB
        - 6 MiB
        - 256 KiB
        - 32 KiB
        - 16 KiB
        - 32 KiB
        - 384 KiB
        - 20 KiB
      *
        - Radeon PRO W7700
        - RDNA3
        - gfx1101
        - 16 GiB
        - 48
        - 32
        - 128 KiB
        - 64 MiB
        - 4 MiB
        - 256 KiB
        - 32 KiB
        - 16 KiB
        - 32 KiB
        - 384 KiB
        - 20 KiB
      *
        - Radeon PRO W6800
        - RDNA2
        - gfx1030
        - 32 GiB
        - 60
        - 32
        - 128 KiB
        - 128 MiB
        - 4 MiB
        - 128 KiB
        - 16 KiB
        - 16 KiB
        - 32 KiB
        - 256 KiB
        - 20 KiB
      *
        - Radeon PRO W6600
        - RDNA2
        - gfx1032
        - 8 GiB
        - 28
        - 32
        - 128 KiB
        - 32 MiB
        - 2 MiB
        - 128 KiB
        - 16 KiB
        - 16 KiB
        - 32 KiB
        - 256 KiB
        - 20 KiB
      *
        - Radeon PRO V620
        - RDNA2
        - gfx1030
        - 32 GiB
        - 72
        - 32
        - 128 KiB
        - 128 MiB
        - 4 MiB
        - 128 KiB
        - 16 KiB
        - 16 KiB
        - 32 KiB
        - 256 KiB
        - 20 KiB
      *
        - Radeon Pro W5500
        - RDNA
        - gfx1012
        - 8 GiB
        - 22
        - 32
        - 128 KiB
        -
        - 4 MiB
        - 128 KiB
        - 16 KiB
        - 16 KiB
        - 32 KiB
        - 256 KiB
        - 20 KiB
      *
        - Radeon Pro VII
        - GCN5.1
        - gfx906
        - 16 GiB
        - 60
        - 64
        - 64 KiB
        -
        - 4 MiB
        -
        - 16 KiB
        - 16 KiB per 3 CUs
        - 32 KiB per 3 CUs
        - 256 KiB
        - 12.5 KiB

.. dropdown:: AMD Radeon GPUs

  .. list-table::
      :header-rows: 1
      :name: radeon-arch-spec-table

      *
        - Model
        - Architecture
        - LLVM target name
        - VRAM
        - Compute Units
        - Warp Size
        - LDS
        - Infinity Cache
        - L2 Cache
        - Graphics L1 Cache
        - L0 Vector Cache
        - L0 Scalar Cache
        - L0 Instruction Cache
        - VGPR File
        - SGPR File
      *
        - Radeon RX 7900 XTX
        - RDNA3
        - gfx1100
        - 24 GiB
        - 96
        - 32
        - 128 KiB
        - 96 MiB
        - 6 MiB
        - 256 KiB
        - 32 KiB
        - 16 KiB
        - 32 KiB
        - 384 KiB
        - 20 KiB
      *
        - Radeon RX 7900 XT
        - RDNA3
        - gfx1100
        - 20 GiB
        - 84
        - 32
        - 128 KiB
        - 80 MiB
        - 6 MiB
        - 256 KiB
        - 32 KiB
        - 16 KiB
        - 32 KiB
        - 384 KiB
        - 20 KiB
      *
        - Radeon RX 7900 GRE
        - RDNA3
        - gfx1100
        - 16 GiB
        - 80
        - 32
        - 128 KiB
        - 64 MiB
        - 6 MiB
        - 256 KiB
        - 32 KiB
        - 16 KiB
        - 32 KiB
        - 384 KiB
        - 20 KiB
      *
        - Radeon RX 7800 XT
        - RDNA3
        - gfx1101
        - 16 GiB
        - 60
        - 32
        - 128 KiB
        - 64 MiB
        - 4 MiB
        - 256 KiB
        - 32 KiB
        - 16 KiB
        - 32 KiB
        - 384 KiB
        - 20 KiB
      *
        - Radeon RX 7700 XT
        - RDNA3
        - gfx1101
        - 12 GiB
        - 54
        - 32
        - 128 KiB
        - 48 MiB
        - 4 MiB
        - 256 KiB
        - 32 KiB
        - 16 KiB
        - 32 KiB
        - 384 KiB
        - 20 KiB
      *
        - Radeon RX 7600
        - RDNA3
        - gfx1102
        - 8 GiB
        - 32
        - 32
        - 128 KiB
        - 32 MiB
        - 2 MiB
        - 256 KiB
        - 32 KiB
        - 16 KiB
        - 32 KiB
        - 256 KiB
        - 20 KiB
      *
        - Radeon RX 6950 XT
        - RDNA2
        - gfx1030
        - 16 GiB
        - 80
        - 32
        - 128 KiB
        - 128 MiB
        - 4 MiB
        - 128 KiB
        - 16 KiB
        - 16 KiB
        - 32 KiB
        - 256 KiB
        - 20 KiB
      *
        - Radeon RX 6900 XT
        - RDNA2
        - gfx1030
        - 16 GiB
        - 80
        - 32
        - 128 KiB
        - 128 MiB
        - 4 MiB
        - 128 KiB
        - 16 KiB
        - 16 KiB
        - 32 KiB
        - 256 KiB
        - 20 KiB
      *
        - Radeon RX 6800 XT
        - RDNA2
        - gfx1030
        - 16 GiB
        - 72
        - 32
        - 128 KiB
        - 128 MiB
        - 4 MiB
        - 128 KiB
        - 16 KiB
        - 16 KiB
        - 32 KiB
        - 256 KiB
        - 20 KiB
      *
        - Radeon RX 6800
        - RDNA2
        - gfx1030
        - 16 GiB
        - 60
        - 32
        - 128 KiB
        - 128 MiB
        - 4 MiB
        - 128 KiB
        - 16 KiB
        - 16 KiB
        - 32 KiB
        - 256 KiB
        - 20 KiB
      *
        - Radeon RX 6750 XT
        - RDNA2
        - gfx1031
        - 12 GiB
        - 40
        - 32
        - 128 KiB
        - 96 MiB
        - 3 MiB
        - 128 KiB
        - 16 KiB
        - 16 KiB
        - 32 KiB
        - 256 KiB
        - 20 KiB
      *
        - Radeon RX 6700 XT
        - RDNA2
        - gfx1031
        - 12 GiB
        - 40
        - 32
        - 128 KiB
        - 96 MiB
        - 3 MiB
        - 128 KiB
        - 16 KiB
        - 16 KiB
        - 32 KiB
        - 256 KiB
        - 20 KiB
      *
        - Radeon RX 6700
        - RDNA2
        - gfx1031
        - 10 GiB
        - 36
        - 32
        - 128 KiB
        - 80 MiB
        - 3 MiB
        - 128 KiB
        - 16 KiB
        - 16 KiB
        - 32 KiB
        - 256 KiB
        - 20 KiB
      *
        - Radeon RX 6650 XT
        - RDNA2
        - gfx1032
        - 8 GiB
        - 32
        - 32
        - 128 KiB
        - 32 MiB
        - 2 MiB
        - 128 KiB
        - 16 KiB
        - 16 KiB
        - 32 KiB
        - 256 KiB
        - 20 KiB
      *
        - Radeon RX 6600 XT
        - RDNA2
        - gfx1032
        - 8 GiB
        - 32
        - 32
        - 128 KiB
        - 32 MiB
        - 2 MiB
        - 128 KiB
        - 16 KiB
        - 16 KiB
        - 32 KiB
        - 256 KiB
        - 20 KiB
      *
        - Radeon RX 6600
        - RDNA2
        - gfx1032
        - 8 GiB
        - 28
        - 32
        - 128 KiB
        - 32 MiB
        - 2 MiB
        - 128 KiB
        - 16 KiB
        - 16 KiB
        - 32 KiB
        - 256 KiB
        - 20 KiB
      *
        - Radeon VII
        - GCN5.1
        - gfx906
        - 16 GiB
        - 60
        - 64
        - 64 KiB per CU
        -
        - 4 MiB
        -
        - 16 KiB
        - 16 KiB per 3 CUs
        - 32 KiB per 3 CUs
        - 256 KiB
        - 12.5 KiB

For a detailed explanation of the terms refer to the
:ref:`specific documents and guides <gpu-arch-documentation>` or the
:ref:`HIP programming guide <HIP:understand/programming_model>`.

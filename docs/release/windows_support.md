# GPU and OS Support (Windows)

(supported_skus)=

## Supported SKUs

AMD ROCm™ Platform supports the following Windows SKU.

| Distribution        |Processor Architectures| Validated update   |
|---------------------|-----------------------|--------------------|
| Windows 10          | x86-64                | 22H2 (GA)          |
| Windows 11          | x86-64                | 22H2 (GA)          |
| Windows Server 2019 | x86-64                | 1809 (LTSC)        |

For more information on Windows versions, refer to
[Windows release health](https://learn.microsoft.com/en-us/windows/release-health/)
.

## GPU Support Table

::::{tab-set}

:::{tab-item} Radeon Pro™
:sync: radeonpro

[Use Radeon Pro Driver](https://www.amd.com/en/support/linux-drivers)

| Name | Architecture | Product ID|[LLVM Target](https://www.llvm.org/docs/AMDGPUUsage.html#processors) | Runtime | SDK (ISV) | All (ROCm datacenter) |
|:----:|:------------:|:-----------|:--------------------------------------------------------------------:|:-------:|:----------------:|:---------:|
| AMD Radeon™ Pro W7900   | RDNA3  | Navi31 | gfx1100 | ✅ | ✅ | ✅ |
| AMD Radeon™ Pro W7800   | RDNA3  | Navi31 | gfx1100 | ✅ | ✅ | ✅ |
| AMD Radeon™ Pro W6800   | RDNA2  | Navi21 | gfx1030 | ✅ | ✅ | ✅ |
| AMD Radeon™ Pro V620    | RDNA2  | Navi21 | gfx1030 | ✅ | ✅ | ✅ |
| AMD Radeon™ Pro W6600   | RDNA2  | Navi21 | gfx1032 | ✅ | ❌ | ❌ |
| AMD Radeon™ Pro W5500   | RDNA1  | Navi12 | gfx1012 | ✅ | ❌ | ❌ |
| AMD Radeon™ Pro VII     | GCN5.1 | Vega20 | gfx906  | ✅ | ✅ | ✅ |
| AMD Radeon™ Pro WX 8200 | GCN5.0 | Vega10 | gfx900  | ⚠️ | ❌ | ❌ |
| AMD Radeon™ Pro Duo     | GCN4.0 | Fiji   | gfx803  | ❌ | ❌ | ❌ |
| AMD Radeon™ Pro WX 5100 | GCN4.0 | Fiji   | gfx803  | ❌ | ❌ | ❌ |

:::

:::{tab-item} Radeon™
:sync: radeon

[Use Radeon Pro Driver](https://www.amd.com/en/support/linux-drivers)

| Name | Architecture | [LLVM Target](https://www.llvm.org/docs/AMDGPUUsage.html#processors) | Runtime | SDK (ISV) | All (ROCm datacenter) |
|:----:|:------------:|:--------------------------------------------------------------------:|:-------:|:----------------:|:---------:|
| AMD Radeon™ RX 7900 XTX | RDNA3  | gfx1100 | ✅ | ✅ | ✅ |
| AMD Radeon™ RX 7900 XT  | RDNA3  | gfx1100 | ✅ | ✅ | ✅ |
| AMD Radeon™ RX 7600 XT  | RDNA3  | gfx1102 | ✅ | ❌ | ❌ |
| AMD Radeon™ RX 7600M XT | RDNA3  | gfx1102 | ✅ | ❌ | ❌ |
| AMD Radeon™ RX 7600M    | RDNA3  | gfx1102 | ✅ | ❌ | ❌ |
| AMD Radeon™ RX 7700S    | RDNA3  | gfx1102 | ✅ | ❌ | ❌ |
| AMD Radeon™ RX 7600S    | RDNA3  | gfx1102 | ✅ | ❌ | ❌ |
| AMD Radeon™ RX 6950 XT  | RDNA2  | gfx1030 | ✅ | ✅ | ❌ |
| AMD Radeon™ RX 6900 XT  | RDNA2  | gfx1030 | ✅ | ✅ | ❌ |
| AMD Radeon™ RX 6800 XT  | RDNA2  | gfx1030 | ✅ | ✅ | ❌ |
| AMD Radeon™ RX 6800     | RDNA2  | gfx1030 | ✅ | ✅ | ❌ |
| AMD Radeon™ RX 6750 XT  | RDNA2  | gfx1031 | ✅ | ❌ | ❌ |
| AMD Radeon™ RX 6700 XT  | RDNA2  | gfx1031 | ✅ | ❌ | ❌ |
| AMD Radeon™ RX 6700     | RDNA2  | gfx1031 | ✅ | ❌ | ❌ |
| AMD Radeon™ RX 6850M    | RDNA2  | gfx1031 | ✅ | ❌ | ❌ |
| AMD Radeon™ RX 6800M    | RDNA2  | gfx1031 | ✅ | ❌ | ❌ |
| AMD Radeon™ RX 6700M    | RDNA2  | gfx1031 | ✅ | ❌ | ❌ |
| AMD Radeon™ RX 6650 XT  | RDNA2  | gfx1032 | ✅ | ❌ | ❌ |
| AMD Radeon™ RX 6600 XT  | RDNA2  | gfx1032 | ✅ | ❌ | ❌ |
| AMD Radeon™ RX 6600     | RDNA2  | gfx1032 | ✅ | ❌ | ❌ |
| AMD Radeon™ RX 6650M XT | RDNA2  | gfx1032 | ✅ | ❌ | ❌ |
| AMD Radeon™ RX 6650M    | RDNA2  | gfx1032 | ✅ | ❌ | ❌ |
| AMD Radeon™ RX 6600M    | RDNA2  | gfx1032 | ✅ | ❌ | ❌ |
| AMD Radeon™ RX 6800S    | RDNA2  | gfx1032 | ✅ | ❌ | ❌ |
| AMD Radeon™ RX 6700S    | RDNA2  | gfx1032 | ✅ | ❌ | ❌ |
| AMD Radeon™ RX 6600S    | RDNA2  | gfx1032 | ✅ | ❌ | ❌ |
| AMD Radeon™ RX 6500 XT  | RDNA2  | gfx1034 | ✅ | ❌ | ❌ |
| AMD Radeon™ RX 6550S    | RDNA2  | gfx1034 | ✅ | ❌ | ❌ |
| AMD Radeon™ RX 6550M    | RDNA2  | gfx1034 | ✅ | ❌ | ❌ |
| AMD Radeon™ RX 6500M    | RDNA2  | gfx1034 | ✅ | ❌ | ❌ |
| AMD Radeon™ RX 6450M    | RDNA2  | gfx1034 | ✅ | ❌ | ❌ |
| AMD Radeon™ RX 6400     | RDNA2  | gfx1034 | ✅ | ❌ | ❌ |
| AMD Radeon™ RX 5700 XT  | RDNA1  | gfx1010 | ✅ | ❌ | ❌ |
| AMD Radeon™ RX 5700     | RDNA1  | gfx1010 | ✅ | ❌ | ❌ |
| AMD Radeon™ RX 5600 XT  | RDNA1  | gfx1010 | ✅ | ❌ | ❌ |
| AMD Radeon™ RX 5500 XT  | RDNA1  | gfx1012 | ✅ | ❌ | ❌ |
| AMD Radeon™ VII         | GCN5.1 | gfx906  | ✅ | ✅ | ✅ |
| AMD Radeon™ RX Vega 64  | GCN5.0 | gfx900  | ⚠️ | ❌ | ❌ |
| AMD Radeon™ RX Vega 45  | GCN5.0 | gfx900  | ⚠️ | ❌ | ❌ |
| AMD Radeon™ RX 590      | GCN4.0 | gfx803  | ❌ | ❌ | ❌ |
| AMD Radeon™ RX 580      | GCN4.0 | gfx803  | ❌ | ❌ | ❌ |
| AMD Radeon™ RX 580x     | GCN4.0 | gfx803  | ❌ | ❌ | ❌ |
| AMD Radeon™ RX 570      | GCN4.0 | gfx803  | ❌ | ❌ | ❌ |
| AMD Radeon™ RX 570X     | GCN4.0 | gfx803  | ❌ | ❌ | ❌ |
| AMD Radeon™ RX 570 X2   | GCN4.0 | gfx803  | ❌ | ❌ | ❌ |
| AMD Radeon™ RX 560 XT   | GCN4.0 | gfx803  | ❌ | ❌ | ❌ |
| AMD Radeon™ RX 480      | GCN4.0 | gfx803  | ❌ | ❌ | ❌ |
| AMD Radeon™ RX 470      | GCN4.0 | gfx803  | ❌ | ❌ | ❌ |
| AMD Radeon™ R9 Fury     | GCN3.0 | gfx803  | ❌ | ❌ | ❌ |
| AMD Radeon™ R9 Fury X   | GCN3.0 | gfx803  | ❌ | ❌ | ❌ |
| AMD Radeon™ R9 Fury X2  | GCN3.0 | gfx803  | ❌ | ❌ | ❌ |
| AMD Radeon™ R9 Nano     | GCN3.0 | gfx803  | ❌ | ❌ | ❌ |

:::

::::

### Component Support

- **Runtime**: Runtime enables the use of the HIP/OpenCL runtimes only.
  [reference](../reference/all) for details on ROCm.
- **Math & Primitive**: these components refer to libraries found under
  [Math Libraries](../reference/gpu_libraries/math.md) and
  [C++ Primitive Libraries](../reference/gpu_libraries/c%2B%2B_primitives.md).
- **AI & HPC**: Windows does not support AI or HPC. These components refer to
  libraries found under
  [Communication Libraries](../reference/gpu_libraries/communication.md),
  [AI Libraries](../reference/ai_tools.md) and
  [Computer Vision](../reference/computer_vision.md).

### Support Status

- ✅: **Supported** - AMD enables these GPUs in our software distributions for
  the corresponding ROCm product.
- ⚠️: **Deprecated** - Support will be removed in a future release.
- ❌: **Unsupported** - This configuration is not enabled in our software
  distributions.

## CPU Support

ROCm requires CPUs that support PCIe™ Atomics. Modern CPUs after the release of
1st generation AMD Zen CPU and Intel™ Haswell support PCIe Atomics.

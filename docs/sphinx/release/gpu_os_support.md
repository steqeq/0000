# GPU and OS Support

## OS Support

ROCm supports the operating systems listed below.
| OS                 | Validated Kernel   |
|:------------------:|:------------------:|
| RHEL 9.1           | `5.14`             |
| RHEL 8.6 to 8.7    | `4.18`             |
| SLES 15 SP4        |                    |
| Ubuntu 20.04.5 LTS | `5.15`             |
| Ubuntu 22.04.1 LTS | `5.15`, OEM `5.17` |

## Virtualization Support

ROCm supports virtualization for select GPUs only as shown below.

| Hypervisor     | Version  | GPU   | Validated Guest OS (validated kernel)                                            |
|:--------------:|:--------:|:-----:|:--------------------------------------------------------------------------------:|
| VMWare         |ESXi 8    | MI250 | `Ubuntu 20.04 (5.15.0-56-generic)`                                               |
| VMWare         |ESXi 8    | MI210 | `Ubuntu 20.04 (5.15.0-56-generic)`, `SLES 15 SP4 (5.14.21-150400.24.18-default)` |
| VMWare         |ESXi 7    | MI210 | `Ubuntu 20.04 (5.15.0-56-generic)`, `SLES 15 SP4( 5.14.21-150400.24.18-default)` |

## GPU Support Table

|GPU               |Architecture    | [LLVM Target](https://www.llvm.org/docs/AMDGPUUsage.html#processors) | [Support Level](#gpu-support-levels) |
|:----------------:|:--------------:|:--------------------------------------------------------------------:|:------------------------------------:|
|Instinct™ MI250X  | CDNA2          | gfx90a                                                               | Full                                 |
|Instinct™ MI250   | CDNA2          | gfx90a                                                               | Full                                 |
|Instinct™ MI210   | CDNA2          | gfx90a                                                               | Full                                 |
|Instinct™ MI100   | CDNA           | gfx908                                                               | Full                                 |
|Instinct™ MI50    | Vega           | gfx906                                                               | Maintenance                          |
|Radeon™ Pro W6800 | RDNA2          | gfx1030                                                              | Full                                 |
|Radeon™ Pro V620  | RDNA2          | gfx1030                                                              | Full                                 |
|Radeon™ RX 6900 XT| RDNA2          | gfx1030                                                              | Non-commercial                       |
|Radeon™ RX 6600   | RDNA2          | gfx1031                                                              | HIP                                  |
|Radeon™ R9 Fury   | Fiji           | gfx803                                                               | Community                            |

### GPU Support Levels

GPU support levels in ROCm:

- Full - AMD provides full support for all software that is part of ROCm
- Non-commercial - AMD enables all software that is part of ROCm. However,
  commercial usage is not supported.
- HIP SDK - AMD supports select GPU libraries and the HIP Runtime on these
  products. The HIP SDK contents are described [here]().
- HIP - AMD supports the HIP Runtime only for these products.
- Maintenance - This GPUs is now in maintenance mode. No new features will be
  enabled on this product.
- Community - Packages distributed by AMD have dropped support for these GPUs or
  never enabled support for the GPUs. Builds from source are not disabled. AMD
  encourages the open source community to enable functionality for these cards.

## CPU Support

ROCm requires CPUs that support PCIe™ Atomics. Modern CPUs after the release of
1st generation AMD Zen CPU and Intel™ Haswell support PCIe Atomics.

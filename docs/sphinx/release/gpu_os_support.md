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
|Instinct™ MI250X  | CDNA2          | gfx90a                                                               |                                      |
|Instinct™ MI250   | CDNA2          | gfx90a                                                               |                                      |
|Instinct™ MI210   | CDNA2          | gfx90a                                                               |                                      |
|Instinct™ MI100   | CDNA           | gfx908                                                               |                                      |
|Instinct™ MI50    | Vega           | gfx906                                                               |                                      |
|Radeon™ Pro W6800 | RDNA2          | gfx1030                                                              |                                      |
|Radeon™ Pro V620  | RDNA2          | gfx1030                                                              |                                      |
|Radeon™ RX 6900 XT| RDNA2          | gfx1030                                                              |                                      |
|Radeon™ RX 6600   | RDNA2          | gfx1031                                                              |                                      |
|Radeon™ R9 Fury   | Fiji           | gfx803                                                               |                                      |

### GPU Support Levels

GPU support levels in ROCm:

- Support level 1
- Support level 2 
- Deprecated
- Unsupported


## CPU Support

ROCm requires CPUs that support PCIe™ Atomics. Modern CPUs after the release of
1st generation AMD Zen CPU and Intel™ Haswell support PCIe Atomics.

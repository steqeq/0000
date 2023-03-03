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

|GPU               |Architecture    |Product|[LLVM Target](https://www.llvm.org/docs/AMDGPUUsage.html#processors) | Linux                                | Windows |
|:----------------:|:--------------:|:----:|:--------------------------------------------------------------------:|:------------------------------------:|:-------:|
|Instinct™ MI250X  | CDNA2          |ROCm |gfx90a                                                               |Supported                                  |Unsupported  |
|Instinct™ MI250   | CDNA2          |ROCm |gfx90a                                                               |Supported                                  |Unsupported  |
|Instinct™ MI210   | CDNA2          |ROCm |gfx90a                                                               |Supported                             |Unsupported   |
|Instinct™ MI100   | CDNA           |ROCm|gfx908                                                               |Supported                             |Unsupported  |
|Instinct™ MI50    | Vega           |ROCm|gfx906                                                               |Supported                             |Unsupported  |
|Radeon™ Pro W6800 | RDNA2          |ROCm |gfx1030                                                              |Supported                            |Supported|
|Radeon™ Pro V620  | RDNA2          |ROCm|gfx1030                                                              |Supported                            |Unsupported|
|Radeon™ RX 6900 XT| RDNA2          |HIP SDK|gfx1030                                                              |Supported                             |Supported|
|Radeon™ RX 6600   | RDNA2          |HIP|gfx1031                                                              |Supported|Supported|
|Radeon™ R9 Fury   | Fiji           |ROCm|gfx803                                                               |Community                            |Unsupported|

### Products in ROCm

- ROCm software product include all software that is part of the ROCm ecosystem. Please see [article](link) for details of ROCm.
- HIP SDK software products includes the HIP Runtime and a selection of GPU libraries for compute. Please see [article](link) for details of HIP SDK.
- HIP software product supported GPUs enable use of the HIP Runtime


### GPU Support Levels

GPU support levels in ROCm:

- Supported
- Unsupported - This configuration 
- Deprecated - support will be removed in a future release. 
- Community


## CPU Support

ROCm requires CPUs that support PCIe™ Atomics. Modern CPUs after the release of
1st generation AMD Zen CPU and Intel™ Haswell support PCIe Atomics.

# Linux support

The AMD ROCm™ Platform supports the following Linux distributions.

| Distribution            |Processor architectures| Validated kernel |
|----------------------|-------------------------|-------------------|
| RHEL 9.1                 | x86-64                         | 5.14                    |
| RHEL 8.6 to 8.7       | x86-64                         | 4.18                    |
| SLES 15 SP4            | x86-64                         |                           |
| Ubuntu 20.04.5 LTS | x86-64                        | 5.15                    |
| Ubuntu 22.04.1 LTS | x86-64                        | 5.15, OEM 5.17   |

## Supported hardware

The following GPUs are supported on Linux distributions:

| Product Name | Architecture | [LLVM Target](https://www.llvm.org/docs/AMDGPUUsage.html#processors) |
|-------------------------------|----------|---------|
| AMD Instinct™ MI250X       | CDNA2  | gfx90a |
| AMD Instinct™ MI250         | CDNA2  | gfx90a |
| AMD Instinct™ MI210         | CDNA2  | gfx90a |
| AMD Instinct™ MI100         | CDNA    | gfx908 |
| AMD Instinct™ MI50           | GCN5.1  | gfx906 |
| AMD Instinct™ MI25           | GCN5.0  | gfx900 |
| AMD Radeon™ Pro W6800 | RDNA2  | gfx1030 |
| AMD Radeon™ Pro V620    | RDNA2  | gfx1030 |
| AMD Radeon™ Pro VII        | GCN5.1  | gfx906  |

For AMD Instinct™ products, use the driver that was shipped with ROCm. For Radeon Pro™ products,
download the appropriate [Radeon Pro Driver](https://www.amd.com/en/support/linux-drivers).

ROCm can be used with CPUs that support PCIe™ Atomics. These include CPUs produced after the
release of 1st generation AMD Zen CPU and Intel™ Haswell.

## Virtualization Support

ROCm supports virtualization for the GPUs listed in the following table.

| Hypervisor     | Version  | GPU   | Validated Guest OS (validated kernel)                                            |
|----------------|----------|-------|----------------------------------------------------------------------------------|
| VMWare         | ESXi 8   | MI250 | Ubuntu 20.04 (`5.15.0-56-generic`)                                               |
| VMWare         | ESXi 8   | MI210 | Ubuntu 20.04 (`5.15.0-56-generic`), SLES 15 SP4 (`5.14.21-150400.24.18-default`) |
| VMWare         | ESXi 7   | MI210 | Ubuntu 20.04 (`5.15.0-56-generic`), SLES 15 SP4 (`5.14.21-150400.24.18-default`) |
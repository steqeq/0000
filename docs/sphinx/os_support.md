# OS Support and Virtualization

## OS Support
ROCm supports the operating systems listed below.
| OS                 | Validated Kernel  |
|--------------------|----------------|
| RHEL 9.1           | 5.14           |
| RHEL 8.6 to 8.7    | 4.18           |
| SLES 15 SP4        |                |
| Ubuntu 20.04.5 LTS | 5.15           |
| Ubuntu 22.04.1 LTS | 5.15, OEM 5.17 |

## Virtualization Support
ROCm supports virtualization for select GPUs only as shown below.

| Hypervisor     | Version  | GPU | Validated Guest OS (validated kernel)|
|---------------|----------|-----|----------------|
| VMWare|ESXi 8|MI250|Ubuntu 20.04 (5.15.0-56-generic)|
| VMWare|ESXi 8|MI210|Ubuntu 20.04 (5.15.0-56-generic), SLES 15 SP4 (5.14.21-150400.24.18-default)|
| VMWare|ESXi 7|MI210|Ubuntu 20.04 (5.15.0-56-generic), SLES 15 SP4( 5.14.21-150400.24.18-default)|

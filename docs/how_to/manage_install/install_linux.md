# Installation

Installing can be done in one of two ways, depending on your preference:

- Using an installer script
- Through your system's package manager

```{attention}
For information on installing ROCm on devices with NVIDIA GPUs, refer to the HIP
Installation Guide.
```

## Installer Script Method

::::::{tab-set}
:::::{tab-item} Ubuntu
:sync: ubuntu

::::{rubric} Downloading the installer script
::::

::::{tab-set}
:::{tab-item} Ubuntu 20.04
:sync: ubuntu-20.04

```shell
sudo apt update
wget https://repo.radeon.com/amdgpu-install/5.4.3/ubuntu/focal/amdgpu-install_5.4.50403-1_all.deb
sudo apt install ./amdgpu-install_5.4.50403-1_all.deb
```

:::
:::{tab-item} Ubuntu 22.04
:sync: ubuntu-22.04

```shell
sudo apt update
wget https://repo.radeon.com/amdgpu-install/5.4.3/ubuntu/jammy/amdgpu-install_5.4.50403-1_all.deb
sudo apt install ./amdgpu-install_5.4.50403-1_all.deb
```

:::
::::
:::::
:::::{tab-item} Red Hat Enterprise Linux
:sync: RHEL

::::{rubric} Downloading the installer script
::::

::::{tab-set}
:::{tab-item} RHEL 8.6
:sync: RHEL-8.6

```shell
sudo yum install https://repo.radeon.com/amdgpu-install/5.4.3/rhel/8.6/amdgpu-install-5.4.50403-1.el8.noarch.rpm
```

:::
:::{tab-item} RHEL 8.7
:sync: RHEL-8.7

```shell
sudo yum install https://repo.radeon.com/amdgpu-install/5.4.3/rhel/8.7/amdgpu-install-5.4.50403-1.el8.noarch.rpm
```

:::
:::{tab-item} RHEL 9.1
:sync: RHEL-9.1

```shell
sudo yum install https://repo.radeon.com/amdgpu-install/5.4.3/rhel/9.1/amdgpu-install-5.4.50403-1.el9.noarch.rpm
```

:::
::::
:::::
:::::{tab-item} Red Hat Enterprise Linux
:sync: RHEL

::::{rubric} Downloading the installer script
::::

::::{tab-set} SUSE Linux Enterprise Server 15
:::{tab-item} Service Pack 4
:sync: SLES15-SP4

```shell
sudo zypper --no-gpg-checks install https://repo.radeon.com/amdgpu-install/5.4.3/sle/15.4/amdgpu-install-5.4.50403-1.noarch.rpm
```

:::
::::
:::::
::::::

### Download and Install the Installer

### Using the Installer Script for Single-version ROCm Installation

### Using Installer Script in Docker

### Using the Installer Script for Multiversion ROCm Installation

## Package Manager Method

### Installing ROCm on Linux Distributions

### Understanding the Release-specific AMDGPU and ROCm Stack Repositories on Linux Distributions

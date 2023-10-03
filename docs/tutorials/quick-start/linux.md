# Linux quick-start installation guide

For a quick summary on installing ROCm on Linux, follow the steps listed on this page. If you
want a more in-depth installation guide, see [Installing ROCm on Linux](../install/linux/index.md).

::::::::{tab-set}
:::::::{tab-item} Ubuntu
:sync: ubuntu

::::::{tab-set}
:::::{tab-item} Ubuntu 22.04
:sync: ubuntu-22.04

::::{tab-set}
:::{tab-item} Package Manager
:sync: ubuntu-22.04-PM


```console
$ sudo mkdir --parents --mode=0755 /etc/apt/keyrings
$ wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null
$ sudo tee /etc/apt/sources.list.d/amdgpu.list <<'EOF'
  deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/latest/ubuntu jammy main
  EOF
$ sudo tee /etc/apt/sources.list.d/rocm.list <<'EOF'
  deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/debian jammy main
  EOF
$ echo -e 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' | sudo tee /etc/apt/preferences.d/rocm-pin-600
$ sudo apt update
$ sudo apt install amdgpu-dkms
$ sudo apt install rocm-hip-libraries
$ sudo reboot
```
:::
:::{tab-item} Install Script
:sync: ubuntu-22.04-IS

```console
$ sudo apt update 
$ wget https://repo.radeon.com/amdgpu-install/5.7/ubuntu/jammy/amdgpu-install_5.7.50700-1_all.deb 
$ sudo apt install ./amdgpu-install_5.7.50700-1_all.deb 
$ sudo amdgpu-install --usecase=graphics,rocm 
```

:::
::::

:::::
:::::{tab-item} Ubuntu 20.04
:sync: ubuntu-20.04

::::{tab-set}
:::{tab-item} Package Manager
:sync: ubuntu-20.04-PM

```console
$ sudo mkdir --parents --mode=0755 /etc/apt/keyrings
$ wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null
$ sudo tee /etc/apt/sources.list.d/amdgpu.list <<'EOF'
  deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/latest/ubuntu focal main
  EOF
sudo tee /etc/apt/sources.list.d/rocm.list <<'EOF'
deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/debian focal main
EOF
$ sudo apt update
$ sudo apt install amdgpu-dkms
$ sudo apt install rocm-hip-libraries
$ sudo reboot
```

:::
:::{tab-item} Install Script
:sync: ubuntu-20.04-IS

```console
$ sudo apt update 
$ wget https://repo.radeon.com/amdgpu-install/5.7/ubuntu/focal/amdgpu-install_5.7.50700-1_all.deb 
$ sudo apt install ./amdgpu-install_5.7.50700-1_all.deb 
$ sudo amdgpu-install --usecase=graphics,rocm 
```

:::
::::

::::::

:::::::

:::::::{tab-item} Red Hat Enterprise Linux
:sync: RHEL

::::::{tab-set}
:::::{tab-item} RHEL 9.2
:sync: RHEL-9.2

::::{tab-set}
:::{tab-item} Package Manager
:sync: RHEL-9.2

```console
$ sudo tee /etc/yum.repos.d/amdgpu.repo <<'EOF' 
  [amdgpu]
  name=amdgpu
  baseurl=https://repo.radeon.com/amdgpu/latest/rhel/9.2/main/x86_64
  enabled=1
  gpgcheck=1
  gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
  EOF
$ sudo tee /etc/yum.repos.d/rocm.repo <<'EOF'
  [rocm]
  name=rocm
  baseurl=https://repo.radeon.com/rocm/rhel9/latest/main
  enabled=1
  priority=50
  gpgcheck=1
  gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
  EOF
$ sudo yum clean all
$ sudo yum install amdgpu-dkms
$ sudo yum install rocm-hip-libraries
$ sudo reboot
```
:::
:::{tab-item} Install Script
:sync: RHEL-9.2

```console
$ sudo yum install https://repo.radeon.com/amdgpu-install/5.7/rhel/9.2/amdgpu-install-5.7.50700-1.el9.noarch.rpm 
$ sudo amdgpu-install --usecase=graphics,rocm 
```

:::
::::



:::::
:::::{tab-item} RHEL 9.1
:sync: RHEL-9.1

::::{tab-set}
:::{tab-item} Package Manager
:sync: RHEL-9.1

```console
$ sudo tee /etc/yum.repos.d/amdgpu.repo <<'EOF' 
  [amdgpu]
  name=amdgpu
  baseurl=https://repo.radeon.com/amdgpu/latest/rhel/9.1/main/x86_64
  enabled=1
  gpgcheck=1
  gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
  EOF
$ sudo tee /etc/yum.repos.d/rocm.repo <<'EOF'
  [rocm]
  name=rocm
  baseurl=https://repo.radeon.com/rocm/rhel9/latest/main
  enabled=1
  priority=50
  gpgcheck=1
  gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
  EOF
$ sudo yum clean all
$ sudo yum install amdgpu-dkms
$ sudo yum install rocm-hip-libraries
$ sudo reboot
```
:::
:::{tab-item} Install Script
:sync: RHEL-9.1

```console
$ sudo yum install https://repo.radeon.com/amdgpu-install/5.7/rhel/9.1/amdgpu-install-5.7.50700-1.el9.noarch.rpm 
$ sudo amdgpu-install --usecase=graphics,rocm 
```

:::
::::

:::::

:::::{tab-item} RHEL 8.8
:sync: RHEL-8.8

::::{tab-set}
:::{tab-item} Package Manager
:sync: RHEL-8.8

```console
$ sudo tee /etc/yum.repos.d/amdgpu.repo <<'EOF' 
  [amdgpu]
  name=amdgpu
  baseurl=https://repo.radeon.com/amdgpu/latest/rhel/8.8/main/x86_64
  enabled=1
  gpgcheck=1
  gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
  EOF
$ sudo tee /etc/yum.repos.d/rocm.repo <<'EOF'
  [rocm]
  name=rocm
  baseurl=https://repo.radeon.com/rocm/rhel8/latest/main
  enabled=1
  priority=50
  gpgcheck=1
  gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
  EOF
$ sudo yum clean all
$ sudo yum install amdgpu-dkms
$ sudo yum install rocm-hip-libraries
$ sudo reboot
```
:::
:::{tab-item} Install Script
:sync: RHEL-8.8
```console
$ sudo yum install https://repo.radeon.com/amdgpu-install/5.7/rhel/8.8/amdgpu-install-5.7.50700-1.el8.noarch.rpm 
$ sudo amdgpu-install --usecase=graphics,rocm 
```

:::
::::

:::::

:::::{tab-item} RHEL 8.7
:sync: RHEL-8.7

::::{tab-set}
:::{tab-item} Package Manager
:sync: RHEL-8.7
```console
$ sudo tee /etc/yum.repos.d/amdgpu.repo <<'EOF' 
  [amdgpu]
  name=amdgpu
  baseurl=https://repo.radeon.com/amdgpu/latest/rhel/8.7/main/x86_64
  enabled=1
  gpgcheck=1
  gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
  EOF
$ sudo tee /etc/yum.repos.d/rocm.repo <<'EOF'
  [rocm]
  name=rocm
  baseurl=https://repo.radeon.com/rocm/rhel8/latest/main
  enabled=1
  priority=50
  gpgcheck=1
  gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
  EOF
$ sudo yum clean all
$ sudo yum install amdgpu-dkms
$ sudo yum install rocm-hip-libraries
$ sudo reboot
```
:::
:::{tab-item} Install Script
:sync: RHEL-8.7

```console
$ sudo yum install https://repo.radeon.com/amdgpu-install/5.7/rhel/8.7/amdgpu-install-5.7.50700-1.el8.noarch.rpm 
$ sudo amdgpu-install --usecase=graphics,rocm 
```

:::
::::
:::::

:::::{tab-item} RHEL 8.6
:sync: RHEL-8.6

::::{tab-set}
:::{tab-item} Package Manager
:sync: RHEL-8.6
```console
$ sudo tee /etc/yum.repos.d/amdgpu.repo <<'EOF' 
  [amdgpu]
  name=amdgpu
  baseurl=https://repo.radeon.com/amdgpu/latest/rhel/8.6/main/x86_64
  enabled=1
  gpgcheck=1
  gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
  EOF
$ sudo tee /etc/yum.repos.d/rocm.repo <<'EOF'
  [rocm]
  name=rocm
  baseurl=https://repo.radeon.com/rocm/rhel8/latest/main
  enabled=1
  priority=50
  gpgcheck=1
  gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
  EOF
$ sudo yum clean all
$ sudo yum install amdgpu-dkms
$ sudo yum install rocm-hip-libraries
$ sudo reboot
```
:::
:::{tab-item} Install Script
:sync: RHEL-8.6
```console
$ sudo yum install https://repo.radeon.com/amdgpu-install/5.7/rhel/8.6/amdgpu-install-5.7.50700-1.el8.noarch.rpm 
$ sudo amdgpu-install --usecase=graphics,rocm 
```

:::
::::
:::::
::::::

::::::

:::::::

:::::::{tab-item} SUSE Linux Enterprise Server
:sync: SLES

::::::{tab-set}
:::::{tab-item} SLES 15.5
:sync: SLES-15.5

::::{tab-set}                   
:::{tab-item} Package Manager   
:sync: SLES-15.5

```console
$ sudo tee /etc/zypp/repos.d/amdgpu.repo <<'EOF'
  [amdgpu]
  name=amdgpu
  baseurl=https://repo.radeon.com/amdgpu/latest/sle/15.5/main/x86_64
  enabled=1
  gpgcheck=1
  gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
  EOF
$ sudo tee /etc/zypp/repos.d/rocm.repo <<'EOF'
  [rocm]
  name=rocm
  baseurl=https://repo.radeon.com/rocm/zyp/zypper
  enabled=1
  priority=50
  gpgcheck=1
  gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
  EOF
$ sudo zypper ref
$ sudo zypper install amdgpu-dkms
$ sudo zypper install rocm-hip-libraries
$ sudo reboot
```
:::                             
:::{tab-item} Install Script    
:sync: SLES-15.5          
```console                      
$ sudo zypper --no-gpg-checks install https://repo.radeon.com/amdgpu-install/5.7/sle/15.5/amdgpu-install-5.7.50700-1.noarch.rpm 
$ sudo amdgpu-install --usecase=graphics,rocm                        
```                            

:::                             
::::                            
:::::
:::::{tab-item} SLES 15.4
:sync: SLES-15.4

::::{tab-set}
:::{tab-item} Package Manager
:sync: SLES-15.4
```console
$ sudo tee /etc/zypp/repos.d/amdgpu.repo <<'EOF'
  [amdgpu]
  name=amdgpu
  baseurl=https://repo.radeon.com/amdgpu/latest/sle/15.4/main/x86_64
  enabled=1
  gpgcheck=1
  gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
  EOF
$ sudo tee /etc/zypp/repos.d/rocm.repo <<'EOF'
  [rocm]
  name=rocm
  baseurl=https://repo.radeon.com/rocm/zyp/zypper
  enabled=1
  priority=50
  gpgcheck=1
  gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
  EOF
$ sudo zypper ref
$ sudo zypper install amdgpu-dkms
$ sudo zypper install rocm-hip-libraries
$ sudo reboot
```
:::
:::{tab-item} Install Script
:sync: SLES-15.4

```console
$ sudo zypper --no-gpg-checks install https://repo.radeon.com/amdgpu-install/5.7/sle/15.4/amdgpu-install-5.7.50700-1.noarch.rpm 
$ sudo amdgpu-install --usecase=graphics,rocm 
```

:::
::::
:::::
::::::

::::::


:::::::
::::::::

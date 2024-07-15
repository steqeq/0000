System settings
===============

This guide reviews system settings that are required to configure the system for
AMD Instinct MI300X accelerators. It is important to ensure a system is
functioning correctly before trying to improve the overall performance. In this
guide, settings discussed mostly ensure proper functionality of the Instinct
based system. Some settings discussed are known to improve performance for most
applications running in a MI300X system. The guide does not describe how to
improve performance for specific applications or workloads. 

The main topics of discussion in this document are:

* System BIOS settings

* Operating system settings

* GRUB

System BIOS settings
--------------------

AMD EPYC 9004-based systems
^^^^^^^^^^^^^^^^^^^^^^^^^^^
For maximum MI300X GPU performance on systems with AMD EPYC™ 9004-series
processors (codename “Genoa”) and AMI System BIOS, the following configuration
of system BIOS settings has been validated. These settings must be used for the
qualification process and should be set as default values in the system BIOS.
Analogous settings for other non-AMI System BIOS providers could be set
similarly. For systems with Intel processors, some settings may not apply or be
available as listed in the following table.

Each row in the table details a setting but the specific location within the
BIOS setup menus may be different, or the option may not be present. 

.. list-table::
   :header-rows: 1

   * - BIOS setting location

     - Parameter

     - Value

     - Comments

   * - Advanced / PCI subsystem settings

     - Above 4G decoding

     - Enabled

     - GPU large BAR support.

   * - Advanced / PCI subsystem settings

     - SR-IOV support

     - Enabled

     - Enable single root IO virtualization.

   * - AMD CBS / GPU common options

     - Global C-state control

     - Auto

     - Global C-states -- do not disable this menu item).

   * - AMD CBS / GPU common options

     - CCD/Core/Thread enablement

     - Accept

     - May be necessary to enable the SMT control menu.

   * - AMD CBS / GPU common options / performance

     - SMT control

     - Disable

     - Set to Auto if the primary application is not compute-bound.

   * - AMD CBS / DF common options / memory addressing

     - NUMA nodes per socket

     - Auto

     - Auto = NPS1. At this time, the other options for NUMA nodes per socket
       should not be used.

   * - AMD CBS / DF common options / memory addressing

     - Memory interleaving

     - Auto

     - Depends on NUMA nodes (NPS) setting.

   * - AMD CBS / DF common options / link

     - 4-link xGMI max speed

     - 32 GBps

     - Auto results in the speed being set to the lower of the max speed the
       motherboard is designed to support and the max speed of the CPU in use.

   * - AMD CBS / NBIO common options

     - IOMMU

     - Enabled

     - 

   * - AMD CBS / NBIO common options

     - PCIe ten bit tag support

     - Auto

     - 

   * - AMD CBS / NBIO common options / SMU common options

     - Determinism control

     - Manual

     - 

   * - AMD CBS / NBIO common options / SMU common options

     - Determinism slider

     - Power

     - 

   * - AMD CBS / NBIO common options / SMU common options

     - cTDP control

     - Manual

     - Set cTDP to the maximum supported by the installed CPU.

   * - AMD CBS / NBIO common options / SMU common options

     - cTDP

     - 400

     - Value in watts.

   * - AMD CBS / NBIO common options / SMU common options

     - Package power limit control

     - Manual

     - Set package power limit to the maximum supported by the installed CPU.

   * - AMD CBS / NBIO common options / SMU common options

     - Package power limit

     - 400

     - Value in watts.

   * - AMD CBS / NBIO common options / SMU common options

     - xGMI link width control

     - Manual

     - Set package power limit to the maximum supported by the installed CPU.

   * - AMD CBS / NBIO common options / SMU common options

     - xGMI force width control

     - Force

     - 

   * - AMD CBS / NBIO common options / SMU common options

     - xGMI force link width

     - 2

     - * 0: Force xGMI link width to x2
       * 1: Force xGMI link width to x8
       * 2: Force xGMI link width to x16

   * - AMD CBS / NBIO common options / SMU common options

     - xGMI max speed

     - Auto

     - Auto results in the speed being set to the lower of the max speed the
       motherboard is designed to support and the max speed of the CPU in use.

   * - AMD CBS / NBIO common options / SMU common options

     - APBDIS

     - 1

     - Disable DF (data fabric) P-states

   * - AMD CBS / NBIO common options / SMU common options

     - DF C-states

     - Auto

     - 

   * - AMD CBS / NBIO common options / SMU common options

     - Fixed SOC P-state

     - P0

     - 

   * - AMD CBS / security

     - TSME

     - Disabled

     - Memory encryption

GRUB settings
-------------

In any modern Linux distribution, the ``/etc/default/grub`` file is used to
configure GRUB. In this file, the string assigned to ``GRUB_CMDLINE_LINUX`` is
the command line parameters that Linux uses during boot.

Appending strings via Linux command line
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is recommended to append the following strings in ``GRUB_CMDLINE_LINUX``:

One important parameter is ``pci=realloc=off``. With this setting Linux is able
to unambiguously detect all GPUs of the MI300X based system because this setting
disables the automatic reallocation of PCI resources. It's used when Single Root
I/O Virtualization (SR-IOV) Base Address Registers (BARs) have not been
allocated by the BIOS. This can help avoid potential issues with certain
hardware configurations.

The ``iommu=pt`` setting enables IOMMU pass-through mode. When in pass-through
mode, the adapter does not need to use DMA translation to the memory, which can
improve performance.

The ``pcie=noats`` setting disables PCIe address translation services globally
and affects any ATS-capable IOMMU driver.

IOMMU is a system specific IO mapping mechanism and can be used for DMA mapping
and isolation. This can be beneficial for virtualization and device assignment
to virtual machines. It is recommended to enable IOMMU support.

For a system that has AMD host CPUs add this to ``GRUB_CMDLINE_LINUX``:

.. code-block:: text

   amd_iommu=on iommu=pt

Otherwise, if the system has Intel host CPUs add this instead to
``GRUB_CMDLINE_LINUX``:

.. code-block:: text

   intel_iommu=on iommu=pt

Update GRUB
^^^^^^^^^^^

Update GRUB to use the modified configuration:

.. code-block:: shell

   sudo grub2-mkconfig -o /boot/grub2/grub.cfg

Note that in some Debian systems, the ``grub2-mkconfig`` comamnd is not found.
Check to see whether ``grub-mkconfig`` is available and in addition check to see
that the version of that is version 2 with the use of the following command:

.. code-block:: shell

   grub-mkconfig -version

Operating system settings
-------------------------

System management
-----------------

In order to optimize the system performance, first the existing system configuration parameters and settings need to be understood. ROCm has some CLI tools that can provide system level information which give hints towards optimizing an user application.

For a complete guide on how to install/manage/uninstall ROCm on Linux, refer to Quick-start (Linux). For verifying that the installation was successful, refer to the post-install instructions and system tools. Should verification fail, consult the System Debugging Guide.

Hardware verification with ROCm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ROCm platform provides tools to query the system structure.

ROCm SMI
''''''''

To query your GPU hardware, use the ``rocm-smi`` command. ROCm SMI lists
GPUs available to your system -- with their device ID and their respective
firmware (or VBIOS) versions.

ROCm Bandwidth Test
'''''''''''''''''''


Acronyms
--------

AMI
  American Megatrends International

APBDIS
  Algorithmic Performance Boost Disable

ATS
  Address Translation Services

BAR
  Base Address Register

BIOS
  Basic Input/Output System

CBS
  Common BIOS Settings

CLI
  Command Line Interace

CPU
  Central Processing Unit

cTDP
  Configurable Thermal Design Power

DDR5
  Double Data Rate 5 DRAM

DF
  Data Fabric

DIMM
  Dual In-line Memory Module

DMA
  Direct Memory Access

DPM
  Dynamic Power Management

GPU
  Graphics Processing Unit

GRUB
  Grand Unified Bootloader

HPC
  High Performance Computing

IOMMU
  Input-Output Memory Management Unit

ISA
  Instruction Set Architecture

LCLK
  Link Clock Frequency

NBIO
  North Bridge Input/Output

NUMA
  Non-Uniform Memory Access

PCI
  Peripheral Component Interconnect

PCIe
  PCI Express

POR
  Power-On Reset

SIMD
  Single Instruction, Multiple Data

SMT
  Simultaneous Multithreading

SMI
  System Management Interface

SOC
  System On Chip

SR-IOV
  Single Root I/O Virtualization

TP
  Tensor Parallelism

TSME
  Transparent Secure Memory Encryption

X2APIC
  Extended Advanced Programmable Interrupt Controller

xGMI
  Inter-chip Global Memory Interconnect 

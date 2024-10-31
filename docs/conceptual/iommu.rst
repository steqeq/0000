.. meta::
   :description: Input-Output Memory Management Unit (IOMMU)
   :keywords: IOMMU, DMA, PCIe, xGMI, AMD, ROCm

****************************************************************
Input-Output Memory Management Unit (IOMMU)
****************************************************************

The I/O Memory Management Unit (IOMMU) provides memory remapping services for I/O devices. It adds support for address translation and system memory access protection on direct memory access (DMA) transfers from peripheral devices. 

The IOMMU's memory remapping services are used to:

* provide private I/O space for devices used in a guest virtual machine.
* prevent unauthorized DMA requests to system memory and to memory-mapped I/O (MMIO).
* help in debugging memory access issues.
* facilitate peer-to-peer DMA.

The IOMMU also provides interrupt remapping, which is used by devices that support multiple interrupts and for interrupt delivery on hardware platforms with a large number of cores.

.. note::

  Devices such as the MI300 accelerators are connected via XGMI links, and don't use PCI/PCIe for peer-to-peer DMA. Because PCI/PCIe is not used for peer-to-peer DMA, there are no device physical addressing limitations or platform root port limitations. However, because non-GPU devices such as RDMA NICs use PCIe for peer-to-peer DMA, there might still be physical addressing limitations and platform root port limitations when these non-GPU devices interact with other devices, including GPUs.

Linux supports IOMMU in both virtualized environments and bare metal. 

The IOMMU is enabled by default but can be disabled or put into passthrough mode through the Linux kernel command line:

.. list-table:: 
  :widths: 25 25 50
  :header-rows: 1

  * - IOMMU Mode
    - Kernel command
    - Description
  * - Enabled
    - default setting
    - The IOMMU is enabled in remapping mode. Each device gets its own I/O virtual address space. All devices on Linux register their DMA addressing capabilities and the kernel will ensure that any address space mapped for DMA is mapped within the device's DMA addressing limits. Only address space explicitly mapped by the devices will be mapped into virtual address space. Attempts to access an unmapped page will generate an IOMMU page fault. This setting is recommended for AMD Radeon GPUs that need peer-to-peer DMA and want to avoid platform-specific addressing limitations.
  * - Passthrough
    - ``iommu=pt``
    - The IOMMU is enabled with interrupt remapping enabled but I/O remapping disabled. The entire platform shares a common platform address space for system memory and MMIO spaces, ensuring compatibility with drivers from external vendors, while still supporting CPUs with a large number of cores. This setting is recommended for AMD Instinct Accelerators and for AMD Radeon GPUs.
  * - Disabled
    - ``iommu=off``
    - The IOMMU is disabled and the entire platform shares a common platform address space for system memory and MMIO spaces.

The IOMMU also provides virtualized access to the MMIO portions of the platform address space for peer-to-peer DMA.

Because peer-to-peer DMA is not officially part of the PCI/PCIe specification, the behavior of peer-to-peer DMA varies between hardware platforms. 

AMD CPUs earlier than AMD Zen only supported peer-to-peer DMA for writes. On CPUs from AMD Zen and onwards, peer-to-peer DMA is fully supported. 

To use peer-to-peer DMA on Linux, the following options must be enabled in your Linux kernel configuration:

* ``CONFIG_PCI_P2PDMA``
* ``CONFIG_DMABUF_MOVE_NOTIFY`` 
* ``CONFIG_HSA_AMD_P2P``

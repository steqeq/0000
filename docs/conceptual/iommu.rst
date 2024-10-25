.. meta::
   :description: Input-Output Memory Management Unit (IOMMU)
   :keywords: IOMMU, DMA, PCIe, XGMI, AMD, ROCm

****************************************************************
Input-Output Memory Management Unit (IOMMU)
****************************************************************

The input-output memory management unit (IOMMU) provides memory remapping services for IO devices. 

Similar to the memory management unit (MMU) on a CPU, the IOMMU is able to provide a private virtual address space for each device or group of devices connected to a CPU. 

The IOMMU's memory remapping services are used in the following ways:

* To provide private IO space for devices used in a guest virtual machine. 
* To prevent unauthorized direct memory access (DMA) requests to the memory-managed IO (MMIO).
* To help in debugging memory access issues.
* To facilitate peer-to-peer DMA.

The IOMMU also provides interrupt remapping, which is useful for devices that support multiple interrupts and for interrupt delivery on hardware platforms with several cores.

Linux supports IOMMU in both virtualized environments and bare metal. 

The IOMMU is enabled by default, but can be disabled or put into passthrough mode through the Linux kernel command line:

.. list-table:: 
  :widths: 25 25 50
  :header-rows: 1

  * - IOMMU Mode
    - Kernel command
    - Description
  * - Enabled
    - default setting
    - The IOMMU is enabled in remapping mode. Each device gets its own IO virtual address space. All devices on Linux register their DMA addressing capabilities and the kernel will ensure that any address space mapped for DMA is mapped within the DMA addressing limits of the device. Only address space explicitly mapped by the devices will be mapped into virtual address space. Attempts to access an unmapped page will generate an IOMMU page fault.
  * - Passthrough
    - ``iommu=pt``
    - The IOMMU is enabled but the entire platform still shares a common platform address space for system memory and MMIO spaces. The passthrough setting is used in cases where some aspects of the IOMMU, such as interrupt remapping, are needed, but memory remapping is not.
  * - Disabled
    - ``iommu=off``
    - The IOMMU is disabled and the entire platform shares a common platform address space for system memory and MMIO spaces.


The IOMMU also provides virtualized access to the MMIO portions of the platform address space for peer-to-peer DMA.

Because peer-to-peer DMA is not officially part of the PCI/PCIe specification, the behavior of peer-to-peer DMA varies between hardware platforms. 

AMD CPUs prior to AMD Zen only supported peer-to-peer DMA for writes. On CPUs from AMD Zen and onwards, peer-to-peer DMA is fully supported. 

To use peer-to-peer DMA on Linux, the following options must be enabled in your Linux kernel configuration:

* ``CONFIG_PCI_P2PDMA``
* ``CONFIG_DMABUF_MOVE_NOTIFY`` 
* ``CONFIG_HSA_AMD_P2P``

Devices connected via XGMI links, such as the MI50/60, MI100, MI200, and MI300 accelerators, have private XGMI links between GPUs and the PCIe bus is not used for peer-to-peer DMA. Because the PCIe bus is not used for peer-to-peer DMA, there are no device physical addressing limitations or platform root port limitations. However, because non-GPU devices such as RDMA NICs use PCIe for peer-to-peer DMA, there would be device physical addressing limitations and platform root port limitations when they interact with GPUs.

.. meta::
   :description: AMD hardware optimization for specific workloads
   :keywords: high-performance computing, HPC, Instinct accelerators, Radeon,
              AMD, ROCm, system, EPYC, CPU, GPU, BIOS, OS

*******************
System optimization
*******************

System administrators can optimize the performance of their AMD hardware
generally and based on specific workloads and use cases. This section outlines
recommended system optimization options for AMD accelerators and GPUs, enabling
administrators to maximize efficiency and performance.

High-performance computing workloads
====================================

High-performance computing (HPC) workloads have unique requirements that may not
be fully met by the default hardware and BIOS configurations of OEM platforms.
To achieve optimal performance for HPC workloads, it is crucial to adjust
settings at both the platform and workload levels. 

The :ref:`AMD Instinct™ accelerator optimization guides <mi-optimization-guides>`
in this section describe:

* BIOS settings that can impact performance
* Hardware configuration best practices
* Supported versions of operating systems
* Workload-specific recommendations for optimal BIOS and operating system
  settings

The guides might also discuss the AMD Instinct software development
environment, including information on how to install and run the DGEMM, STREAM,
HPCG, and HPL benchmarks. The guides provide a good starting point but is
not tested exhaustively across all compilers.

Knowledge prerequisites to better understand the following
:ref:`Instinct system optimization guides <mi-optimization-guides>` and to
perform tuning for HPC applications include:

* Experience in configuring servers
* Administrative access to the server's Management Interface (BMC)
* Administrative access to the operating system
* Familiarity with the OEM server's BMC (strongly recommended)
* Familiarity with the OS specific tools for configuration, monitoring, and
  troubleshooting (strongly recommended)

While the following guides are a good starting point, developers are encouraged
to perform their own performance testing for additional tuning per device and
per workload.

.. _mi-optimization-guides:

.. list-table::
   :header-rows: 1
   :stub-columns: 1

   * - System optimization guide

     - Architecture reference

     - White papers

   * - :doc:`AMD Instinct MI300X <system-optimization/mi300x>`

     - `AMD Instinct MI300 instruction set architecture <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf>`_

     - `CDNA 3 architecture <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf>`_

   * - :doc:`AMD Instinct MI200 <system-optimization/mi200>`

     - `AMD Instinct MI200 instruction set architecture <https://www.amd.com/system/files/TechDocs/instinct-mi200-cdna2-instruction-set-architecture.pdf>`_

     - `CDNA 2 architecture <https://www.amd.com/system/files/documents/amd-cdna2-white-paper.pdf>`_

   * - :doc:`AMD Instinct MI100 <system-optimization/mi100>`

     - `AMD Instinct MI100 instruction set architecture <https://www.amd.com/system/files/TechDocs/instinct-mi100-cdna1-shader-instruction-set-architecture%C2%A0.pdf>`_

     - `CDNA architecture <https://www.amd.com/system/files/documents/amd-cdna-whitepaper.pdf>`_

Workstation workloads
=====================

Workstation workloads, much like those for HPC, have a unique set of
requirements: a blend of both graphics and compute, certification, stability and
others.

The document covers specific software requirements and processes needed to use
these GPUs for Single Root I/O Virtualization (SR-IOV) and machine learning
tasks.

The main purpose of this document is to help users utilize the RDNA™ 2 GPUs to
their full potential.

.. _rdna-optimization-guides:

.. list-table::
   :header-rows: 1
   :stub-columns: 1

   * - System optimization guide

     - Architecture reference

     - White papers

   * - :doc:`AMD Radeon PRO W6000 and V620 <system-optimization/w6000-v620>`

     - `AMD RDNA 2 instruction set architecture <https://www.amd.com/system/files/TechDocs/rdna2-shader-instruction-set-architecture.pdf>`_

     - `RDNA 2 architecture <https://www.amd.com/system/files/documents/rdna2-explained-radeon-pro-W6000.pdf>`_


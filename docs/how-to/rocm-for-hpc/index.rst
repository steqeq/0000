.. meta::
   :description: How to use ROCm for HPC
   :keywords: ROCm, AI, high performance computing, HPC, usage, tutorial

******************
Using ROCm for HPC
******************

The ROCm™ open-source software stack is optimized to extract the best high
performance computing (HPC) workload performance from AMD Instinct™ accelerators
while maintaining compatibility with industry software frameworks.

For more information, see :doc:`What is ROCm? <../../what-is-rocm>`.

Some of the most popular HPC frameworks are part of the ROCm platform, including
those to help parallelize operations across multiple acclerators and servers,
handle memory hierarchies, and solve linear systems.

Our catalog of GPU-accelerated applications includes a vast set of
platform-compatible HPC applications, including those in astrophysics, climate 
and weather, computational chemistry, computational fluid dynamics, earth
science, genomics, geophysics, molecular dynamics, and physics. Refer to the
resources in the following table for ready-to-install build instructions and
deployment suggestions for AMD Instinct accelerators.

.. raw:: html

   <style>
     ul {
       padding: 0;
       list-style: none;
     }
   </style>

.. list-table::
   :header-rows: 1

   * - Application domain
     - HPC applications

   * - Physics
     - 
       * `Chroma <https://github.com/amd/InfinityHub-CI/tree/main/chroma/`_
       * `Grid <https://github.com/amd/InfinityHub-CI/tree/main/grid/`_
       * `MILC <https://github.com/amd/InfinityHub-CI/tree/main/milc/`_
       * `PIConGPU <https://github.com/amd/InfinityHub-CI/tree/main/picongpu`_

   * - Astrophysics
     - `Cholla <https://github.com/amd/InfinityHub-CI/tree/main/cholla/`_

   * - Geophysics
     - `Specfrem3D-Cartesian <https://github.com/amd/InfinityHub-CI/tree/main/specfem3d>`_

   * - Molecular dynamics
     - 
       * `Gromacs with HIP (AMD implementation) <https://github.com/amd/InfinityHub-CI/tree/main/gromacs>`_
       * `LAMMPS <https://github.com/amd/InfinityHub-CI/tree/main/lammps>`_

   * - Computational fluid dynamics
     -
       * `NEKO <https://github.com/amd/InfinityHub-CI/tree/main/neko>`_
       * `nekRS <https://github.com/amd/InfinityHub-CI/tree/main/nekrs>`_

   * - Computational chemistry
     - `QUDA <https://github.com/amd/InfinityHub-CI/tree/main/quda>`_
   
   * - Quantum Monte Carlo Simulation
     - `QMCPACK <https://github.com/amd/InfinityHub-CI/tree/main/qmcpack>`_

   * - Electronic structure
     - `CP2K <https://github.com/amd/InfinityHub-CI/tree/main/cp2k>`_

   * - Climate and weather
     - `MPAS <https://github.com/amd/InfinityHub-CI/tree/main/mpas>`_

   * - Benchmarking
     -
       * HPCG
       * rocHPL
       * rocHPL-MxP

   * - Tools and libraries
     -
       * ROCm with GPU-aware MPI container
       * Kokkos
       * PyFR
       * RAJA
       * Trilinos

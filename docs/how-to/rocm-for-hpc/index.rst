.. meta::
   :description: How to use ROCm for HPC
   :keywords: ROCm, AI, high performance computing, HPC, usage, tutorial

******************
Using ROCm for HPC
******************

The ROCm open-source software stack is optimized to extract the best high
performance computing (HPC) workload performance from AMD Instinct™ accelerators
while maintaining compatibility with industry software frameworks.

ROCm enhances support and access for developers by providing streamlined and
improved tools that significantly increase productivity. Being open-source, ROCm
fosters innovation, differentiation, and collaboration within the developer
community, making it a powerful and accessible solution for leveraging the full
potential of AMD accelerators' capabilities in diverse computational
applications.

* For more information, see :doc:`What is ROCm? <../../what-is-rocm>`.

* For guidance on installing ROCm, see :doc:`rocm-install-on-linux:index`. See
  the :doc:`../../compatibility/compatibility-matrix` for details on hardware
  and operating system support.

Some of the most popular HPC frameworks are part of the ROCm platform, including
those to help parallelize operations across multiple accelerators and servers,
handle memory hierarchies, and solve linear systems.

.. image:: ../../data/how-to/rocm-for-hpc/hpc-stack-2024_6_20.png
   :align: center
   :alt: Software and hardware ecosystem surrounding ROCm and AMD Instinct for HPC

The following catalog of GPU-accelerated solutions includes a vast set of
platform-compatible HPC applications, including those for astrophysics, climate 
and weather, computational chemistry, computational fluid dynamics, earth
science, genomics, geophysics, molecular dynamics, and physics computing. Refer
to the resources in the following table for instructions on building, running,
and deployment using AMD Instinct accelerators.

.. _hpc-apps:

..
   Reduce font size of HPC app descriptions slightly.

.. raw:: html

   <style>
     #hpc-apps-table tr td:last-child {
       font-size: 0.9rem;
     }
   </style>

.. container::
   :name: hpc-apps-table

   .. list-table::
      :header-rows: 1
      :stub-columns: 1
      :widths: 2 2 5

      * - Application domain
        - HPC application
        - Description

      * - Physics
        - `Chroma <https://github.com/amd/InfinityHub-CI/tree/main/chroma/>`_
        - Data-parallel programming constructs for lattice field theory -- lattice
          quantum chromodynamics (QCD) in particular. It's used to simulate the
          behavior of quarks and gluons. Chroma uses SciDAC QDP++ data-parallel
          programming (in C++) that presents a single high-level code image to
          the user, but can generate highly optimized code for many
          architectural systems including single node workstations, multi and
          many-core nodes, clusters of nodes via QMP, and classic vector
          computers.

      * -
        - `Grid <https://github.com/amd/InfinityHub-CI/tree/main/grid/>`_
        - Library for lattice QCD calculations that uses a high-level data
          parallel approach and techniques to target multiple types of
          parallelism. Grid currently supports MPI, OpenMP and short vector
          parallelism. The code requires at least one AMD accelerator to run.

      * -
        - `MILC <https://github.com/amd/InfinityHub-CI/tree/main/milc/>`_
        - Set of research codes developed by the MIMD Lattice Computation (MILC)
          collaboration for doing simulations of four-dimensional SU(3) lattice
          gauge theory on MIMD parallel machines scaling from single-processor
          workstations to HPC systems.

      * -
        - `PIConGPU <https://github.com/amd/InfinityHub-CI/tree/main/picongpu>`_
        - Particle-in-Cell code that runs on AMD accelerators. The PIC algorithm
          is a central tool in plasma physics. It describes the dynamics of a
          plasma by computing the motion of electrons and ions in the plasma
          based on the Vlasov-Maxwell system of equations. 

      * - Astrophysics
        - `Cholla <https://github.com/amd/InfinityHub-CI/tree/main/cholla/>`_
        - Static-mesh, GPU-native hydrodynamics simulation code that efficiently
          runs high-resolution simulations on massively-parallel computers.
          Cholla is written in a combination of C++, :doc:`HIP <hip:index>`, and
          CUDA C, and requires at least one AMD accelerator to run.

      * - Geophysics
        - `SPECFEM3D Cartesian <https://github.com/amd/InfinityHub-CI/tree/main/specfem3d>`_
        - Simulates acoustic (fluid), elastic (solid), coupled acoustic/elastic,
          poroelastic or seismic wave propagation in any type of conforming mesh
          of hexahedra (structured or otherwise). It can, for instance, model
          seismic waves propagating in sedimentary basins or any other regional
          geological model following earthquakes. It is also used for
          non-destructive testing and for ocean acoustics.

      * - Molecular dynamics
        - `GROMACS with HIP (AMD implementation) <https://github.com/amd/InfinityHub-CI/tree/main/gromacs>`_
        - Recipe based on an
          `AMD fork of the GROMACS project <https://github.com/ROCm/gromacs>`_
          optimized for AMD accelerators. GROMACS is a versatile package to
          perform molecular dynamics simulations of systems with hundreds to
          millions of particles.

      * -
        - `LAMMPS <https://github.com/amd/InfinityHub-CI/tree/main/lammps>`_
        - Large-scale Atomic/Molecular Massively Parallel Simulators: a
          classical molecular dynamics library with a focus on materials
          modeling. It is capable of modeling 2D and 3D systems composed of a
          few up to billions of particles using AMD Instinct accelerators via
          the Kokkos backend.

      * - Computational fluid dynamics
        - `NEKO <https://github.com/amd/InfinityHub-CI/tree/main/neko>`_
        - Portable framework written in modern Fortran for high-order spectral
          element flow simulations. Using an object-oriented design, Neko allows
          for multi-tier abstraction for solver stacks and allows Neko to be
          built against various types of hardware backends.

      * -
        - `nekRS <https://github.com/amd/InfinityHub-CI/tree/main/nekrs>`_
        - Open-source Navier-Stokes solver based on the spectral element method
          targeting CPUs and accelerators that support :doc:`HIP <hip:index>`,
          CUDA, or OpenCL. 

      * - Computational chemistry
        - `QUDA <https://github.com/amd/InfinityHub-CI/tree/main/quda>`_
        - Library designed for efficient lattice QCD computations on
          accelerators. It includes optimized Dirac operators and a variety of
          fermion solvers and conjugate gradient (CG) implementations, enhancing
          performance and accuracy in lattice QCD simulations.

      * - Electronic structure
        - `CP2K <https://github.com/amd/InfinityHub-CI/tree/main/cp2k>`_
        - Versatile code for atomistic simulations across various systems:
          solid state, liquid, molecular, periodic, material, crystal, and
          biological. It supports multiple theory levels including DFTB, LDA,
          GGA, MP2, RPA, semi-empirical methods, and classical force fields.
          CP2K enables simulations such as molecular dynamics, metadynamics,
          Monte Carlo, Ehrenfest dynamics, vibrational analysis, core level
          spectroscopy, energy minimization, and transition state optimization
          using NEB or dimer methods.

      * - Quantum Monte Carlo Simulation
        - `QMCPACK <https://github.com/amd/InfinityHub-CI/tree/main/qmcpack>`_
        - Quantum Monte Carlo code designed for computing the electronic
          structure of atoms, molecules, 2D nanomaterials, and solids. It
          supports a wide range of materials, including metallic systems and
          insulators. QMCPACK is optimized to perform effectively across various
          computing platforms, from workstations to the latest supercomputers.
          In addition to high performance, QMCPACK prioritizes code quality and
          reproducibility.

      * - Climate and weather
        - `MPAS <https://github.com/amd/InfinityHub-CI/tree/main/mpas>`_
        - Collaborative project between COSIM at Los Alamos and the National
          Center for Atmospheric Research for developing atmosphere, ocean, and
          other Earth-system simulation components for use in climate, regional
          climate, and weather studies.

      * - Benchmark
        - `HPCG <https://github.com/amd/InfinityHub-CI/tree/main/hpcg>`_
        - High Performance Conjugate Gradient Benchmark: a complement to the
          High Performance LINPACK (HPL) benchmark. The computational and data
          access patterns of HPCG are designed to closely match a broad set of
          important applications not represented by HPL, and to incentivize
          computer system designers to invest in capabilities that benefit the
          collective performance of these applications.

      * -
        - `rocHPL <https://github.com/amd/InfinityHub-CI/tree/main/rochpl>`_
        - Implementation of the High Performance LINPACK (HPL) benchmark on the
          ROCm platform: a benchmark which solves a uniformly random system of
          linear equations and reports floating-point execution rate.

      * -
        - `rocHPL-MxP <https://github.com/amd/InfinityHub-CI/tree/main/hpl-mxp>`_
        - Benchmark that highlights the convergence of HPC and AI workloads by
          solving a system of linear equations using novel, mixed-precision
          algorithms.

      * - Tools and libraries
        - `ROCm with GPU-aware MPI container <https://github.com/amd/InfinityHub-CI/tree/main/base-gpu-mpi-rocm-docker>`_
        - Base container for GPU-aware MPI with ROCm for HPC applications. This
          project provides boilerplate for building and running a Docker
          container with ROCm supporting GPU-aware MPI implementations using
          either OpenMPI or UCX.

      * -
        - `Kokkos <https://github.com/amd/InfinityHub-CI/tree/main/kokkos>`_
        - C++ programming model for writing performant portable applications for
          use across HPC platforms. It provides abstractions for both parallel
          execution of code and data management. Kokkos targets complex node
          architectures with N-level memory hierarchies and multiple types of
          execution resources.

      * -
        - `PyFR <https://github.com/amd/InfinityHub-CI/tree/main/pyfr>`_
        - Open-source Python framework for solving advection-diffusion type
          problems on streaming architectures using the flux reconstruction
          approach (Huynh). PyFR solves a range of governing systems on mixed
          unstructured grids containing various element types. It's designed to
          target a range of hardware platforms via an in-built domain-specific
          language derived from the Mako templating engine.

      * -
        - `RAJA <https://github.com/amd/InfinityHub-CI/tree/main/raja>`_
        - Library of C++ software abstractions that enables architecture and
          programming model portability for HPC applications. RAJA is primarily
          developed at Lawrence Livermore National Laboratory (LLNL).

      * -
        - `Trilinos <https://github.com/amd/InfinityHub-CI/tree/main/trilinos>`_
        - Portable toolkit for scientific computing. Trilinos is built on top of
          the Kokkos portability layer. So, it has support for all manner of
          architectures using a MPI+X methodology where MPI handles
          communication between distributed memory spaces, and local compute can
          be handled using a variety of CPU and GPU parallelization APIs such as
          :doc:`HIP <hip:index>`, OpenMP, CUDA, and others, all of which are
          abstracted away by Kokkos.

To learn about ROCm for AI applications, see :doc:`../rocm-for-ai/index`.

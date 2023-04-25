***********
Using CMake
***********

Most components in ROCm support CMake. Porjects depnding on header-only or
library components typically require CMake 3.5 or higher whereas those wanting
to make use of CMake's HIP language support will require CMake 3.21 or higher.

Most components in ROCm support detection, consumption by CMake builds. They do
so via shipping config files searched by CMake's ``find_package`` command. ROCm
also supports driving CMake's HIP language.

Most components in ROCm support CMake 3.5 or higher out-of-the-box and do not
require any special Find modules. A Find module is often used by downstream to
find the files by guessing locations of files with platform-specific hints.
Typically, the Find module is required when the upstream is not built with CMake
or the package configuration files are not available.

ROCm provides the respective *config-file* packages, and this enables
``find_package`` to be used directly. ROCm does not require any Find module as
the *config-file* packages are shipped with the upstream projects.

Finding Dependencies
====================

.. note::
   For a complete
   reference on how to deal with dependencies in CMake, refer to the CMake docs
   on `find_package
   <https://cmake.org/cmake/help/latest/command/find_package.html>`_ and the
   `Using Dependencies Guide
   <https://cmake.org/cmake/help/latest/guide/using-dependencies/index.html>`_
   to get an overview of CMake's related facilities.

In short, CMake supports finding dependencies in two ways:

-  In Module mode, it consults a file ``Find<PackageName>.cmake`` which tries to
   find the component in typical install locations and layouts. CMake ships a
   few dozen such scripts, but users and projects may ship them as well.
-  In Config mode, it locates a file named ``<packagename>-config.cmake`` or
   ``<PackageName>Config.cmake`` which describes the installed component in all
   regards needed to consume it.

ROCm dominantly relies on Config mode, one notable exception being the Module
driving the compilation of HIP programs on Nvidia runtimes. As such, when
dependencies are not found in standard system locations, one either has to
instruct CMake to search for package config files in additional folders using
the ``CMAKE_PREFIX_PATH`` variable (a semi-colon separated list of filesystem
paths), or using ``<PackageName>_ROOT`` variable on a project-specific basis.

There are nearly a dozen ways to set these variables. One may be more convenient
over the other depending on your workflow. Conceptually the simplest is adding
it to your CMake configuration command on the command-line via
``-D CMAKE_PREFIX_PATH=....`` . AMD packaged ROCm installs can typically be
added to the config file search paths such as:

-  Windows: ``-D CMAKE_PREFIX_PATH=${env:HIP_PATH}``

-  Linux: ``-D CMAKE_PREFIX_PATH=/opt/rocm``

For a complete guide on where and how ROCm may be installed on a system, refer
to the installation guides in these docs (Windows, Linux).

Using HIP in CMake
==================

ROCm componenents providing a C/C++ interface support being consumed using any
C/C++ toolchain that CMake knows how to drive. ROCm also supports CMake's HIP
language features, allowing users to program using the HIP single-source
programming model. The HIP API without compiling GPU device code behaves as a
C/C++ library.

Consuming ROCm C/C++ Libraries
------------------------------

Libraries such as rocBLAS, rocFFT, MIOpen, etc. behave as C/C++ libraries.
Illustrated in the example below is a C++ application using MIOpen from CMake.
It calls ``find_package(miopen)``, which provides the ``MIOpen`` imported
target. This can be linked with ``target_link_libraries``::

    project(myProj LANGUAGES CXX)
    find_package(miopen)
    add_library(myLib ...)
    target_link_libraries(myLib PUBLIC MIOpen)

.. note::
    Most libraries are designed as host-only API, so using a GPU device
    compiler is not necessary for downstream projects unless they use GPU device
    code.

Consuming the HIP API
---------------------

-  Use the HIP API without compiling the GPU device code. As there is no GPU
   code, any C or C++ compiler can be used. The ``find_package(hip)`` provides
   the ``hip::host`` target to use HIP in this context

::

   project(myProj)
   find_package(hip REQUIRED)
   add_executable(myApp ...)
   target_link_libraries(myApp PRIVATE hip::host)

.. note::
    The ``hip::host`` target provides all the usage requirements needed to use
    HIP without compiling GPU device code.

-  Use HIP API and compile GPU device code. This requires using a
   device compiler. The compiler for CMake can be set using either the
   ``CMAKE_C_COMPILER`` and ``CMAKE_CXX_COMPILER`` variable or using the ``CC``
   and ``CXX`` environment variables. This can be set when configuring CMake or
   put into a CMake toolchain file. The device compiler must be set to a
   compiler that supports AMD GPU targets, which is usually Clang.

The ``find_package(hip)`` provides the ``hip::device`` target to add all the
flags for device compilation

::

  # Search for rocm in common locations
  list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
  # Find hip
  find_package(hip)
  # Create library
  add_library(myLib ...)
  # Link with HIP
  target_link_libraries(myLib hip::device)

This project can then be configured with

::

    cmake -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ ..

Which uses the device compiler provided from the binary packages from
`repo.radeon.com <http://repo.radeon.com>`_.

.. note::
    Compiling for the GPU device requires at least C++11. This can be
    enabled by setting ``CMAKE_CXX_STANDARD`` or setting the correct compiler flags
    in the CMake toolchain.

The GPU device code can be built for different GPU architectures by
setting the ``GPU_TARGETS`` variable. By default, this is set to all the
currently supported architectures for AMD ROCm. It can be set by passing
the flag during configuration with ``-DGPU_TARGETS=gfx900``. It can also be
set in the CMakeLists.txt as a cached variable before calling
``find_package(hip)``::

    # Set the GPU to compile for
    set(GPU_TARGETS "gfx900" CACHE STRING "GPU targets to compile for")
    # Search for rocm in common locations
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
    # Find hip
    find_package(hip)


ROCm CMake Packages
--------------------

+-----------+----------+--------------------------------------------------------+
| Component | Package  | Targets                                                |
+===========+==========+========================================================+
| HIP       | hip      | ``hip::host``, ``hip::device``                         |
+-----------+----------+--------------------------------------------------------+
| rocPRIM   | rocprim  | ``roc::rocprim``                                       |
+-----------+----------+--------------------------------------------------------+
| rocThrust | rocthrust| ``roc::rocthrust``                                     |
+-----------+----------+--------------------------------------------------------+
| hipCUB    | hipcub   | ``hip::hipcub``                                        |
+-----------+----------+--------------------------------------------------------+
| rocRAND   | rocrand  | ``roc::rocrand``                                       |
+-----------+----------+--------------------------------------------------------+
| rocBLAS   | rocblas  | ``roc::rocblas``                                       |
+-----------+----------+--------------------------------------------------------+
| rocSOLVER | rocsolver| ``roc::rocsolver``                                     |
+-----------+----------+--------------------------------------------------------+
| hipBLAS   | hipblas  | ``roc::hipblas``                                       |
+-----------+----------+--------------------------------------------------------+
| rocFFT    | rocfft   | ``roc::rocfft``                                        |
+-----------+----------+--------------------------------------------------------+
| hipFFT    | hipfft   | ``hip::hipfft``                                        |
+-----------+----------+--------------------------------------------------------+
| rocSPARSE | rocsparse| ``roc::rocsparse``                                     |
+-----------+----------+--------------------------------------------------------+
| hipSPARSE | hipsparse| ``roc::hipsparse``                                     |
+-----------+----------+--------------------------------------------------------+
| rocALUTION|rocalution| ``roc::rocalution``                                    |
+-----------+----------+--------------------------------------------------------+
| RCCL      | rccl     | ``rccl``                                               |
+-----------+----------+--------------------------------------------------------+
| MIOpen    | miopen   | ``MIOpen``                                             |
+-----------+----------+--------------------------------------------------------+
| MIGraphX  | migraphx | ``migraphx::migraphx``, ``migraphx::migraphx_c``,      |
|           |          | ``migraphx::migraphx_cpu``, ``migraphx::migraphx_gpu``,|
|           |          | ``migraphx::migraphx_onnx``, ``migraphx::migraphx_tf`` |
+-----------+----------+--------------------------------------------------------+
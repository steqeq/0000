.. meta::
    :description: HIP programming guide
    :keywords: HIP, heterogeneous programming, ROCm API, AMD GPU programming

.. _hip-programming-guide:

********************************************************************************
HIP programming guide
********************************************************************************

ROCm provides a robust environment for heterogeneous programs running on CPUs
and AMD GPUs. ROCm supports a variety of programming languages and frameworks to
help developers access the power of AMD GPUs. Currently supported programming
languages include HIP (Heterogeneous-Compute Interface for Portability) and OpenCL,
as well as languages based on wrappers such as Python, and Fortran. 

HIP is an API based on C++ that provides a runtime and libraries for GPU
programming, and is the key programming language in ROCm. HIP is also designed
to be a marshalling language, allowing code written for NVIDIA's CUDA to be
easily ported to run on AMD GPUs. Developers can use HIP to write kernels that
execute on AMD GPUs while maintaining compatibility with CUDA-based systems.

OpenCL (Open Computing Language) is an open standard for cross-platform,
parallel programming of diverse processors. ROCm supports OpenCL for developers
who want to use standard frameworks across different hardware platforms,
including CPUs, GPUs, and other accelerators. For more information, see `OpenCL <https://www.khronos.org/api/index_2017/opencl>`_.

ROCm/HIP also has python bindings that can be found at https://github.com/ROCm/hip-python.
In modern use cases, Python with TensorFlow and PyTorch is popular due to its
role in AI and Machine Learning.

For a complete description of the HIP programming language, see the :doc:`HIP documentation <hip:index>`.

.. meta::
  :description: This chapter describes the C++ support of the HIP ecosystem
  ROCm software.
  :keywords: AMD, ROCm, HIP, C++

*******************************************************************************
C++ support
*******************************************************************************

.. _language_introduction:
Introduction
===============================================================================
The main way to harness the power of the ROCm platform is through the usage C++ and HIP
code. This code is then compiled with a ``clang`` or ``clang++`` compiler. The official
versions support the HIP platform, but for the most up to date feature set use the
``amdclang`` or ``amdclang++`` installed with your ROCm installation.

The source code will be processed based on the ``C++03``, ``C++11``, ``C++14``, ``C++17``
or ``C++20`` standards, but supports HIP specific extentions and is subject to certain
restrictions. The largest restriction is the lack of standard library support. This is
mostly because the SIMD nature of the HIP device makes most of the standard library
implementations not performant or useful. The important operations are implemented in
HIP specific libraries like rocPRIM, rocThrust and hipCUB.

.. _language_c++11_support:
C++11 support
===============================================================================
The C++11 standard introduced a miriad of new features to the language. These features
are supported in HIP device code, with some notable omissions. The biggest of which is
the lack of concurrency support on the device. This is because the HIP device concurrency
model isfundamentaly different compared to the C++ model, which is used on the host side.
E.G: it doesn't make sense to start a new thread on the device.
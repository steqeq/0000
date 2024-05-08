.. meta::
  :description: This chapter describes the usage of ROCgdb, a tool for debugging
  ROCm software.
  :keywords: AMD, ROCm, HIP, ROCgdb, performance, debugging

*******************************************************************************
ROCgdb
*******************************************************************************

.. _rocgdb_introduction:
Introduction
===============================================================================
This document introduces ROCgdb, the AMD ROCm debugger for Linux targets.

ROCgdb is an extension to GDB, the GNU Project debugger. The tool provides developers
with a mechanism for debugging CUDA applications running on actual hardware. This
enables developers to debug applications without the potential variations introduced
by simulation and emulation environments. It is meant to present a seamless debugging
environment that allows simultaneous debugging of both GPU and CPU code within the
same application, just like programming in HIP is a seamless extension of C++
programming. The existing GDB debugging features are inherently present for debugging
the host code, and additional features have been provided to support debugging ROCm
device code.

ROCgdb supports HIP kernel debugging. It allows the user to set breakpoints, to 
single-step ROCm applications, and also to inspect and modify the memory and variables
of any given thread running on the hardware.

.. _rocgdb_installation:
Installation
===============================================================================
The ROCm installation also installs the ROCgdb debugger, but some steps are necessary
before using the debugger.

.. _rocgdb_setup:
Setup
===============================================================================
Before debugging you have to compile your software with debug information. To do this
you have to add the '-g' flag for your compilation command. This will generate debug
information even when optimizations are turned on. Notice that higher optimization
levels make the debugging more difficult, so it might be useful to turn off these
optimizations by using the '-O0' compiler option.

.. _rocgdb_debugging:
Debugging
===============================================================================
This section is a brief introduction on how to use ROCgdb. For a more information on the
functionality of gdb look up the gdb documentation.

First step is to run ROCgdb with your ROCm application:

``rocgdb my_application``

At this point the application is not running, but you'll have access to the debugger
console. Here you can use every gdb option for host debugging and you can use them and
extra ROCgdb specific features for device debugging.

Before you run your application with the debugger, you'll need to set a breakpoint.

``tbreak my_app.cpp:458``

This will place a breakpoint at the specified line. To start your application use this
command:

``run``

If the breakpoint is in device code the debugger will show the device and host threads as
well. The device threads will not be individual threads, instead they represent a
wavefront on the device. You can switch between the device wavefronts, like you would
between host threads.

You can also switch between layouts. Use different layouts for different situations while
debugging.

``layout src``

``layout asm``

The `src` layout is the source code view, while the `asm` is the assembly view. There are
further layouts you can look up on the gdb documentation.

``info threads``

This command lists all threads with id and information on where the thread is stopped.

To switch threads you can use the following command:

``thread <id>``

To take a step in the execution use:

``n``

To dump the content of the current wavefronts registers use:

``i r``

For further information on the usage of gdb, you can go to the `gdb documentation
<https://www.sourceware.org/gdb/documentation/>`_.

Using GPU Sanitizer

The LLVM Address Sanitizer provides a process that allows developers to detect runtime addressing errors in applications and libraries. The detection is achieved using a combination of compiler-added instrumentation and runtime techniques, including function interception and replacement.

Until now, the LLVM Address Sanitizer process was only available for traditional purely CPU applications. However, ROCm has extended this mechanism to additionally allow the detection of some addressing errors on the GPU in heterogeneous applications. Ideally, developers should treat heterogeneous HIP and OpenMP applications exactly like pure CPU applications. However, this simplicity has not been achieved yet.

This document provides documentation on using ROCm Address Sanitizer.
For information about LLVM Address Sanitizer, see https://clang.llvm.org/docs/AddressSanitizer.html.

Compiling Address Sanitizer
The address sanitizer process begins by compiling the application of interest with the address sanitizer instrumentation.

Recommendations for doing this are:

Compile as many application and dependent library sources as possible using an AMD-built clang-based compiler such as `amdclang++``.

Add the following options to the existing compiler and linker options:
`-fsanitize=address` - enables instrumentation)
`-shared-libsan` - use shared version of runtime)
`-g` - add debug info for improved reporting)

Explicitly use `xnack+` in the offload architecture option. For example, `--offload-arch=gfx90a:xnack+`
Other architectures are allowed, but their device code will not be instrumented and a warning will be emitted.

It is not an error to compile some files without address sanitizer instrumentation, but doing so reduces the ability of the process to detect addressing errors. However, if the main program "a.out" does not directly depend on the Address Sanitizer runtime (`libclang_rt.asan-x86_64.so`) after the build completes (check by running `ldd`,`readelf``), the application will immediately report an error at runtime as described in the next section.

About Compilation Time
When `-fsanitize=address` is used, the LLVM compiler adds instrumentation code around every memory operation. This added code must be handled by all of the downstream components of the compiler toolchain and results in increased overall compilation time. This increase is especially evident in the AMDGPU device compiler and has in a few instances raised the compile time to an unacceptable level.

There are a few options if the compile time becomes unacceptable:

- Avoid instrumentation of the files which have the worst compile times. This will reduce the effectiveness of the address sanitizer process.
- Add the option `-fsanitize-recover=address`` to the compiles with the worst compile times. This option simplifies the added instrumentation resulting in faster compilation. See below for more information.
- Disable instrumentation on a per-function basis by adding `__attribute__`((no_sanitize("address"))) to functions found to be responsible for the large compile time. Again, this will reduce the effectiveness of the process.

Using AMD Supplied Address Sanitizer Instrumented Libraries
ROCm releases have optional packages containing additional address sanitizer instrumented builds of the ROCm libraries usually found in /opt/rocm-<version>/lib. The instrumented libraries have identical names as the regular uninstrumented libraries and are located in /opt/rocm-<version>/lib/asan.
These additional libraries are built using the `amdclang++` and `hipcc` compilers, while some uninstrumented libraries are built with g++. The preexisting build options are used, but, as descibed above, additional options are used: `-fsanitize=address`, `-shared-libsan` and `-g`.

These additional libraries avoid additional developer effort to locate repositories, identify the correct branch, check out the correct tags, and other efforts needed to build the libraries from the source. And they extend the ability of the process to detect addressing errors into the ROCm libraries themselves.

When adjusting an application build to add instrumentation, linking against these instrumented libraries is unnecessary. For example, any `-L` /opt/rocm-<version>/lib compiler options need not be changed. However, the instrumented libraries should be used when the application is run. It is particularly important that the instrumented language runtimes, like `libamdhip64.so` and `librocm-core.so`, are used; otherwise, device invalid access detections may not be reported.

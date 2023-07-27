Using GPU Sanitizer

The LLVM Address Sanitizer provides a process that allows developers to detect runtime addressing errors in applications and libraries. The detection is achieved using a combination of compiler-added instrumentation and runtime techniques, including function interception and replacement.

Until now, the LLVM Address Sanitizer process was only available for traditional purely CPU applications. However, ROCm has extended this mechanism to additionally allow the detection of some addressing errors on the GPU in heterogeneous applications. Ideally, developers should treat heterogeneous HIP and OpenMP applications exactly like pure CPU applications. However, this simplicity has not been achieved yet.

This document provides documentation on using ROCm Address Sanitizer.
For information about LLVM Address Sanitizer, see https://clang.llvm.org/docs/AddressSanitizer.html.

Compiling Address Sanitizer
The address sanitizer process begins by compiling the application of interest with the address sanitizer instrumentation.

Recommendations for doing this are:

Compile as many application and dependent library sources as possible using an AMD-built clang-based compiler such as amdclang++.

Add the following options to the existing compiler and linker options:

`-fsanitize=address` (enables instrumentation)
`-shared-libsan` (use shared version of runtime)
`-g` (add debug info for improved reporting)

Explicitly use `xnack+` in the offload architecture option. For example, `--offload-arch=gfx90a:xnack+`
Other architectures are allowed, but their device code will not be instrumented and a warning will be emitted.

It is not an error to compile some files without address sanitizer instrumentation, but doing so reduces the ability of the process to detect addressing errors. However, if the main program "`a.out`" does not directly depend on the Address Sanitizer runtime (`libclang_rt.asan-x86_64.so`) after the build completes (check by running `ldd` or `readelf`), the application will immediately report an error at runtime as described in the next section.

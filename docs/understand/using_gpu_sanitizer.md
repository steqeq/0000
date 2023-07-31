Using GPU Sanitizer

The LLVM Address Sanitizer provides a process that allows developers to detect runtime addressing errors in applications and libraries. The detection is achieved using a combination of compiler-added instrumentation and runtime techniques, including function interception and replacement.

Until now, the LLVM Address Sanitizer process was only available for traditional purely CPU applications. However, ROCm has extended this mechanism to additionally allow the detection of some addressing errors on the GPU in heterogeneous applications. Ideally, developers should treat heterogeneous HIP and OpenMP applications exactly like pure CPU applications. However, this simplicity has not been achieved yet.

This document provides documentation on using ROCm Address Sanitizer.
For information about LLVM Address Sanitizer, see https://clang.llvm.org/docs/AddressSanitizer.html.

Compiling Address Sanitizer
The address sanitizer process begins by compiling the application of interest with the address sanitizer instrumentation.

Recommendations for doing this are:

Compile as many application and dependent library sources as possible using an AMD-built clang-based compiler such as `amdclang++`.

Add the following options to the existing compiler and linker options:
`-fsanitize=address` - enables instrumentation)
`-shared-libsan` - use shared version of runtime)
`-g` - add debug info for improved reporting)

Explicitly use `xnack+` in the offload architecture option. For example, `--offload-arch=gfx90a:xnack+`
Other architectures are allowed, but their device code will not be instrumented and a warning will be emitted.

It is not an error to compile some files without address sanitizer instrumentation, but doing so reduces the ability of the process to detect addressing errors. However, if the main program "a.out" does not directly depend on the Address Sanitizer runtime (`libclang_rt.asan-x86_64.so`) after the build completes (check by running `ldd`,`readelf`), the application will immediately report an error at runtime as described in the next section.

About Compilation Time
When `-fsanitize=address` is used, the LLVM compiler adds instrumentation code around every memory operation. This added code must be handled by all of the downstream components of the compiler toolchain and results in increased overall compilation time. This increase is especially evident in the AMDGPU device compiler and has in a few instances raised the compile time to an unacceptable level.

There are a few options if the compile time becomes unacceptable:

- Avoid instrumentation of the files which have the worst compile times. This will reduce the effectiveness of the address sanitizer process.
- Add the option `-fsanitize-recover=address` to the compiles with the worst compile times. This option simplifies the added instrumentation resulting in faster compilation. See below for more information.
- Disable instrumentation on a per-function basis by adding `__attribute__`((no_sanitize("address"))) to functions found to be responsible for the large compile time. Again, this will reduce the effectiveness of the process.

Using AMD Supplied Address Sanitizer Instrumented Libraries
ROCm releases have optional packages containing additional address sanitizer instrumented builds of the ROCm libraries usually found in `/opt/rocm-<version>/lib`. The instrumented libraries have identical names as the regular uninstrumented libraries and are located in `/opt/rocm-<version>/lib/asan`.
These additional libraries are built using the `amdclang++` and `hipcc` compilers, while some uninstrumented libraries are built with g++. The preexisting build options are used, but, as descibed above, additional options are used: `-fsanitize=address`, `-shared-libsan` and `-g`.

These additional libraries avoid additional developer effort to locate repositories, identify the correct branch, check out the correct tags, and other efforts needed to build the libraries from the source. And they extend the ability of the process to detect addressing errors into the ROCm libraries themselves.

When adjusting an application build to add instrumentation, linking against these instrumented libraries is unnecessary. For example, any `-L` `/opt/rocm-<version>/lib` compiler options need not be changed. However, the instrumented libraries should be used when the application is run. It is particularly important that the instrumented language runtimes, like `libamdhip64.so` and `librocm-core.so`, are used; otherwise, device invalid access detections may not be reported.

Running Address Sanitizer Instrumented Applications

Preparing to Run an Instrumented Application
Here are a few recommendations to consider before running an address sanitizer instrumented heterogeneous application.

- Ensure the Linux kernel running on the system has Heterogeneous Memory Management (HMM) support. A kernel version of 5.6 or higher should be sufficient.

- Ensure XNACK is enabled
For gfx90a (MI-2X0) or gfx940 (MI-3X0) use environment HSA_XNACK = 1.
For gfx906 (MI-50) or gfx908 (MI-100) use environment HSA_XNACK = 1 but also ensure the amdgpu kernel module is loaded with module argument noretry=0.
This requirement is due to the fact that the XNACK setting for these GPUs is system-wide.

- Ensure that the application will use the instrumented libraries when it runs. The output from the shell command ldd <application name> can be used to see which libraries will be used.
If the instrumented libraries are not listed by ldd, the environment variable LD_LIBRARY_PATH may need to be adjusted, or in some cases an RPATH compiled into the applicaiton may need to be changed and the application recompiled.

- Ensure that the application depends on the address sanitizer runtime. This can be checked by running the command readelf -d <application name> | grep NEEDED and verifying that Shared library: [libclang_rt.asan-x86_64.so] appears in the output.
If it does not appear, when executed the application will quickly output an address sanitizer error that looks like:
==3210==ASan runtime does not come first in initial library list; you should either link runtime to your application or manually preload it with LD_PRELOAD.

- Ensure that the application llvm-symbolizer can be executed, and that it is located in /opt/rocm-<version>/llvm/bin. This executable is not strictly required, but if found is used to translate ("symbolize") a host-side instruction address into a more useful function name, file name, and line number (assuming the application has been built to include debug information).

There is an environment variable, ASAN_OPTIONS which can be used to adjust the runtime behavior of the ASAN runtime itself. There are more than a hundred "flags" that can be adjusted (see an old list at flags) but the default settings are correct and should be used in most cases. It must be noted that these options only affect the host ASAN runtime. The device runtime only currently supports the default settings for the few relevant options.

There are two ASAN_OPTION flags of particular note.

- `halt_on_error=0/1 default 1``.
This tells the ASAN runtime to halt the application immediately after detecting and reporting an addressing error. The default makes sense because the application has entered the realm of undefined behavior. If the developer wishes to have the application continue anyway, this option can be set to zero. However, the application and libraries should then be compiled with the additional option -fsanitize-recover=address. Note that the ROCm optional address sanitizer instrumented libraries are not compiled with this option and if an error is detected within one of them, but halt_on_error is set to 0, more undefined behavior will occur.

- `detect_leaks=0/1 default 1``.
This option directs the address sanitizer runtime to enable the Leak Sanitizer (LSAN). Unfortunately, for heterogeneous applications, this default will result in significant output from the leak sanitizer when the application exits due to allocations made by the language runtime which are not considered to be to be leaks. This output can be avoided by adding detect_leaks=0 to the ASAN_OPTIONS, or alternatively by producing an LSAN suppression file (syntax described here) and activating it with environment variable LSAN_OPTIONS=suppressions=/path/to/suppression/file. When using a suppression file, a suppression report is printed by default. The suppression report can be disabled by using the LSAN_OPTIONS flag print_suppressions=0.

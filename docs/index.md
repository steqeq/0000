# AMD ROCm™ documentation

Our documentation follows the [Diátaxis](https://diataxis.fr/) approach and is divided into four main
categories:

::::{grid} 1 2 2 2
:class-container: rocm-doc-grid

:::{grid-item-card}
:padding: 2
**[Conceptual](conceptual/index.md)**

Topic overviews and background information
^^^

- [Compiler Disambiguation](understand/compiler_disambiguation)
- [Using CMake](understand/cmake_packages)
- [Linux Folder Structure Reorganization](understand/file_reorg)
- [GPU Isolation Techniques](understand/gpu_isolation)
- [GPU Architecture](understand/gpu_arch)

:::

:::{grid-item-card}
:padding: 2
**[How-to](how_to/index.md)**

Task-oriented walkthroughs
^^^

- [System Tuning for Various Architectures](how_to/tuning_guides/index)
- [GPU Aware MPI](how_to/gpu_aware_mpi)
- [Setting up for Deep Learning with ROCm](how_to/deep_learning_rocm)
  - [Magma Installation](how_to/magma_install/magma_install)
  - [PyTorch Installation](how_to/pytorch_install/pytorch_install)
  - [TensorFlow Installation](how_to/tensorflow_install/tensorflow_install)
- [System Level Debugging](how_to/system_debugging.md)

:::

:::{grid-item-card}
:padding: 2
**[Reference](reference/index.md)**

Collated information
^^^

- [Compilers and Development Tools](reference/compilers)
- [HIP](reference/hip)
- [OpenMP](reference/openmp/openmp)
- [Math Libraries](reference/gpu_libraries/math)
- [C++ Primitives Libraries](reference/gpu_libraries/c++_primitives)
- [Communication Libraries](reference/gpu_libraries/communication)
- [AI Libraries](reference/ai_tools)
- [Computer Vision](reference/computer_vision)
- [Management Tools](reference/management_tools)
- [Validation Tools](reference/validation_tools)

:::

:::{grid-item-card}
:padding: 2
**[Tutorials](tutorials/index.md)**

Lesson-oriented material
^^^

- [Installing ROCm](tutorials/install/index.md)
- [Examples](https://github.com/amd/rocm-examples)
- [ML, DL, and AI](examples/machine_learning/all)
  - [](examples/machine_learning/pytorch_inception)
  - [](examples/machine_learning/migraphx_optimization)

:::
::::

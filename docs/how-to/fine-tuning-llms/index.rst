LLM fine-tuning overview
========================

Large Language Models (LLMs) are trained on massive amounts of text
data. As a result, they can generate coherent and fluent text. The
transformer architecture is the fundamental building block of all LLMs
with Transformers, which serves as the underlying architecture that
enables LLMs to understand and generate text by capturing contextual
relationships and long-range dependencies. To better understand the
philosophy of Transformer architecture, review the
`Attention <https://arxiv.org/pdf/1706.03762.pdf>`__\ `is all you
need <https://arxiv.org/pdf/1706.03762.pdf>`__ paper.

By further training on pre-trained LLMs, the new fine-tuned model can
learn knowledge related to specific fields or tasks, thereby
significantly improving its performance in that field or task. The core
idea of fine-tuning is to use the parameters of the pre-trained model as
the starting point for new tasks and shape it through a small amount of
specific domain or task data, enabling the original models' capability
to new tasks or datasets.

Fine-tuning can effectively improve the performance of existing
pre-trained models in specific application scenarios. Continuous
training and adjustment of the parameters of the base model in the
target domain or task can better capture the semantic characteristics
and patterns in specific scenarios, thereby significantly improving the
key indicators of the model in that domain or task. For example, by
fine-tuning the Llama 2 model, the performance in certain functions can
be better than the original language model.

The challenge of fine-tuning models
-----------------------------------

However, the computational cost of fine-tuning is still high, especially
for complex models and large datasets, which poses distinctive
challenges related to substantial computational and memory requirements.
This might be a barrier for low computing power GPUs or GPUs with
limited device memory resources.

For example, suppose we have a language model with 7B parameters,
represented by a weight matrix *W*. During backpropagation, the model
needs to learn a *ΔW* matrix, which updates the original weights to
minimize the value of the loss function.

The weight update is as follows: *W_updated = W + ΔW*.

If the weight matrix *W* contains 7B parameters, then the weight update
matrix *ΔW* should also contain 7B parameters. Hence, the *ΔW*
calculation is computationally and memory intensive.

Optimizations for model fine-tuning
-----------------------------------

A Low-Rank Adaption (LoRA) method for LLM fine-tuning was proposed to
overcome the issue of intensive memory consumption issue. LoRA
introduces a striking solution allowing fast and cost-effective
fine-tuning of state-of-the-art LLMs. This breakthrough ability
accelerates the adjustment process and reduces related memory costs. To
be precise, LoRA decomposes the portion of weight changes *ΔW* into
high-precision low-rank representations, which do not require the
calculations of all *ΔW*. On the contrary, LoRA learns the decomposition
representation of *ΔW* during training, as shown in Figure 1, is the
secret of LoRA in saving computing resources.

LoRA has been integrated into the Hugging Face Parameter-Efficient
Fine-Tuning (PEFT) library, as well as other computation and memory
efficiency optimization variants for model fine-tuning such as AdaLoRA.
This library efficiently adapts large pre-trained models to various
downstream applications without fine-tuning all model parameters. PEFT
methods only fine-tune a few model parameters, significantly decreasing
computational and storage costs, yielding performance comparable to a
fully fine-tuned model. PEFT has been integrated with the Transformers
library, providing a faster and easier way to load, train, and use large
models for inference.

To simplify running a fine-tuning implementation, the Transformer
Reinforcement Learning (TRL) library provides a set of tools to train
transformer language models with reinforcement learning, from the
Supervised Fine-tuning step (SFT), Reward Modeling step (RM), to the
Proximal Policy Optimization (PPO) step. The SFTTrainer in TRL
encapsulates the above PEFT optimizations so users can easily import
their custom training configuration and run the training process.

To demonstrate the benefits of LoRA and the ideal compute compatibility
of using PEFT and TRL libraries on AMD ROCm compatible GPUs, we
conducted a comprehensive implementation on the fine-tuning process of
Llama 2 7B model using LoRA tailored specifically for
question-and-answer tasks on AMD MI300X accelerators. Before starting,
review the key components that form the foundation of this discussion:

-  Llama 2: Meta developed and publicly released a large language model
   family. Its variants range in scale from 7 billion to 70 billion
   parameters.

-  Fine-tuning: A critical process that refines LLM for specialized
   tasks and optimizes the performance.

-  LoRA: a memory-efficient implementation of LLM fine-tuning that
   significantly reduces the number of trainable parameters.

-  SFTTrainer: an optimized trainer with a simple interface to easily
   fine-tune pre-trained models with PEFT adapters, for example, LoRA,
   for memory efficiency purposes on a custom dataset.

Model fine-tuning
=================

This section shows the fine-tuning method for a single GPU device. See
`Multiple GPU model fine-tuning and
inference <#multiple-gpu-model-fine-tuning-and-inference>`__ for
multiple GPUs.

System setup
------------

-  Hardware: AMD Instinct MI300X

-  Software: ROCm 6.1, Ubuntu 22.04, PyTorch 2.1.2, Python 3.10

-  Libraries: transformers, datasets, huggingface-hub, peft, trl, scipy

-  Base model: "meta-llama/Llama-2-7b-chat-hf"

Getting Started 
---------------

1. Setup a base implementation environment

You can pull a prebuilt PyTorch 2.1.2 with ROCm 6.1 (or later) docker
image and launch a docker container by the following command. You need
to change $HOME/ROCM_APP to your local environment.

2. In the docker container, check the availability of ROCm-compatible
   GPUs:

Your output should look like this:

Check if GPUs have been settled properly as the PyTorch backend.

Your output should look like this:

3. Install the required dependencies.

4. Check if the required packages can be imported.

Download base model and fine-tuning dataset 
-------------------------------------------

You need to request to access to download the official Meta Llama model
from Hugging Face. After it is granted, you need to log in with the
following command with personal tokens:

You can also use the NousResearch Llama-2-7b-chat-hf as a substitute.
This has the same model weights as the original.

Run the code to load the based model and tokenizer.

The next step is to fine-tune the base model for a question-and-answer
task using a small data set
called `mlabonne/guanaco-llama2-1k <https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k>`__,
which is a subset (1,000 samples) of
the `timdettmers/openassistant-guanaco <https://huggingface.co/datasets/OpenAssistant/oasst1>`__ data
set. After downloading the base model and data set for fine-tuning, you
can start fine-tuning.

Configure fine-tuning parameters 
--------------------------------

To set up the SFTTrainer parameters, you can use the following code as a
reference.

Start fine-tuning 
-----------------

This step sets twto options. Option 1 sets SFTTrainer with PEFT LoRA,
while Option 2 does not. The section estimates and compares the number
of trainable parameters and training time under the two different
configurations.

Option 1: Training with LoRA 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configure the LoRA:

The output should look like this:

Initialize SFTTrainer with PEFT LoRA Config and Run the trainer:

The output should look like this:

Option 2: Training without LoRA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The output should look like this:

Continue the process using the following code:

The output should look like this:

Save adapters or fully fine-tuned models
----------------------------------------

PEFT methods freeze the pre-trained model parameters during fine-tuning
and add a smaller number of trainable parameters, namely the adapters,
on top of it. The adapters are trained to learn specific task
information. The adapters trained with PEFT are usually an order of
magnitude smaller than the full base model, making them convenient to
share, store, and load.

To save a PEFT adapter once the fine-tuning is completed:

Or if there is no PEFT LoRA configuration for training

The saved PEFT adapter should look like this:

While the saved new full model should look like this:

Note that PEFT adapters can’t be loaded by *AutoModelForCausalLM* from
transformers library as they do not contain full model parameters and
model configurations, for example, config.json. To use it as a normal
transformer model, you will need to merge them into the base model, the
details will be given in the model section for inference.

**
**

Model inference 
===============

The trained model can be classified into three types:

-  Pre-trained language models in Hugging Face

-  Fully fine-tuned models without the use of PEFT

-  PEFT Adapters

This section provides the general methods for model inference on a
single MI300X GPU. Note the implementation environment is based on the
setup for model fine-tuning.

Use pre-trained or fully fine-tuned models
------------------------------------------

If you have a fully fine-tuned model, without using PEFT, you can load
it like any other pre-trained language model in Hugging Face hub using
the transformers library.

In addition, pipelines from transformers offer simple APIs to use
pre-trained models for different tasks, including sentiment analysis,
feature extraction, question answering and so on. You can use the
pipeline abstraction to achieve model inference easily.

Use PEFT adapters
-----------------

To use PEFT adapters like a normal transformer model, you can run the
generation by loading a base model along with PEFT adapters as:

PEFT library provides a **merge_and_unload** method, which merges the
adapter layers into the base model. This is needed if someone wants to
save the adapted model into local storage and use it as a normal
standalone model.

Multiple GPU model fine-tuning and inference 
============================================

.. _system-setup-1:

System setup
------------

-  Hardware: 4 AMD Instinct MI300X GPUs

-  Software: ROCm 6.1, Ubuntu 22.04, PyTorch 2.1.2, Python 3.10

-  Libraries: transformers, datasets, accelerate, huggingface-hub, peft,
   trl, scipy

-  Base model: "meta-llama/Llama-2-7b-chat-hf"

Setup a base implementation environment 
---------------------------------------

Refer to the same setup outlined in `Use pre-trained or fully fine-tuned
models <#use-pre-trained-or-fully-fine-tuned-models>`__. In the docker
container, check the availability of ROCm-compatible Instinct GPUs:

Your output should look like this:

Check if these GPUs have been settled properly as the PyTorch backend.

Your output should look like this:

Use Hugging Face Accelerate 
---------------------------

Hugging Face Accelerate is a library that simplifies turning raw PyTorch
code for a single GPU into code for multiple GPUs for LLM fine-tuning
and inference. It is integrated with Transformers allowing users to
write fully general PyTorch code at scale.

As a brief example of model fine-tuning and inference using multiple
GPUs, use transformers and loading in the Meta llama2 7b model.

1. Model fine-tuning

The same code introduced in Section 1 will be applied for model
fine-tuning on multiple GPUs. The only difference is revising the model
loading using the **device_map** option:

You can let Accelerate handle the device map computation by setting
device_map to one of the supported options ("auto", "balanced",
"balanced_low_0", "sequential") or create one yourself if you want more
control over where each layer should go. The device_map parameter is
recommended to be set to “auto” to allow Accelerate library to
automatically and efficiently allocate the model given the available
resources (4 GPUs in this case). After loading the model, the initial
steps to prepare it have been completed, and the model is fully ready to
use all the resources.

2. Model Inference

During training and inference, you can check the memory usage by running
the rocm-smi command in a terminal. This command produces the following
output showing that all GPUs are involved:

When you have more GPU memory available than the model size, here is the
difference between each option:

-  "auto" and "balanced" evenly split the model on all available GPUs,
   making it possible for you to use a batch size greater than 1.

-  "balanced_low_0" evenly splits the model on all GPUs except the first
   one, and only puts on GPU 0 what does not fit on the others. This
   option is great when you need to use GPU 0 for some processing of the
   outputs, like when using the generate function for Transformers
   models.

"sequential" will fit what it can on GPU 0, then move on GPU 1 and so
forth. Not all GPUs might be used.

Use torchtune Library
---------------------

torchtune is a PyTorch-native library for easy single/multi-GPU model
fine-tuning and inference with LLMs.

The output should look like this:

torchtune recipes are designed around easily composable components and
hackable training loops, with minimal abstraction getting in the way of
fine-tuning your fine-tuning. Run tune ls to show torchtune built-in
RECIPE and CONFIG:

The RECIPE column shows the easy-to-use and hackable
fine-tuning/inference recipes for popular fine-tuning techniques (such
as LoRA). The CONFIG column shows the YAML configs for easily
configuring training, evaluation, quantization, or inference recipes.

The architecture of a YAML config file is shown as:

The following file defines the fine-tuning base model path, data set,
hyper-parameters for optimizer and scheduler, and training data type. To
download the base model for fine-tuning, run the following command:

The output directory after --output-dir should map the model path
specified in YAML config file. To custom the data set,

To launch lora_finetune_distributed on four devices, run the following
command:

Model quantization techniques
=============================

Quantization reduces the model size compared to its native
full-precision version, making it easier to fit large models onto GPUs
with limited memory usage. This section will show you how to run LLM
quantization by using GPTQ, AWQ and bitsandbyes on AMD Instinct
hardware.

GPTQ
----

GPTQ is a post-training quantization technique where each row of the
weight matrix is quantized independently to find a version of the
weights that minimizes the error. These weights are quantized to int4
but are restored to fp16 on the fly during inference. This can save your
memory usage by four times. A speedup in inference is expected because
inference of GPTQ models uses a lower bitwidth, which takes less time to
communicate. Before setting up the GPTQConfig in transformers, ensure
the AutoGPTQ library is installed.

Installing AutoGPTQ
~~~~~~~~~~~~~~~~~~~

The AutoGPTQ library implements the GPTQ algorithm. To install the
latest stable release of AutoGPTQ from pip:

To install AutoGPTQ from source for the ROCm (for example, ROCm 6.1):

Run pip show auto-gptq to show the information of the installed
auto-gptq package. The output should look like this:

Use GPTQ with AutoGPTQ 
~~~~~~~~~~~~~~~~~~~~~~

The examples should be a list of dict whose keys can only be "input_ids"
and "attention_mask".

Setup the quantization configurations:

Load the un-quantized model using AutoGPTQ class and run the
quantization:

Use GPTQ with Transformers
~~~~~~~~~~~~~~~~~~~~~~~~~~

To perform a GPTQ quantization in the Hugging Face Transformers library,
you need to create a GPTQConfig instance and set the number of bits to
quantize to, and a dataset to calibrate the weights. Make sure the
AutoGPTQ library has already been installed.

Load a model to quantize using AutoModelForCausalLM and pass the
gptq_config to its from_pretained method. Set device_map=”auto” to
automatically offload the model to available GPU resources.

Once the model is quantized, you can push the model and tokenizer to
Hugging Face Hub for easy share and access.

Or you can save the model locally:

Exllama-v2 support 
~~~~~~~~~~~~~~~~~~

ExLlama is a Python/C++/CUDA implementation of the Llama model that is
designed for faster inference with 4-bit GPTQ weights. The ExLlama
kernel is activated by default when users create a GPTQConfig object. To
boost inference speed even further on Instinct GPUs, use the ExLlamaV2
kernels by configuring the exllama_config parameter as:

AWQ 
---

Activation-aware Weight Quantization (AWQ) doesn’t quantize all the
weights in a model. Instead, it preserves a small percentage of weights
important for LLM performance. This significantly reduces quantization
loss so you can run models with 4-bit precision without experiencing
performance degradation.

Installing AutoAWQ
~~~~~~~~~~~~~~~~~~

AutoAWQ is a library for quantizing models using the AWQ algorithm.
Users can use the read-to-install wheels from `AutoAWQ
Github <https://github.com/casper-hansen/AutoAWQ/releases/tag/v0.2.4>`__.

To install AutoAWQ from source:

Run pip show autoawq to show the information of the installed AutoAWQ
package. The output should look like this:

Use AWQ with AutoAWQ 
~~~~~~~~~~~~~~~~~~~~

Load the un-quantized model using **awq** class and run the
quantization:

AWQ model Inference with Transformers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Transformers library supports loading AWQ quantized models and
performing generation.

.. _exllama-v2-support-1:

Exllama-v2 support 
~~~~~~~~~~~~~~~~~~

Recent versions of AutoAWQ support exllama-v2 kernels for faster prefill
and decoding.

bitsandbytes
------------

`ROCm bitsandbytes <https://github.com/ROCm/bitsandbytes>`__ library is
a lightweight Python wrapper around CUDA custom functions, in particular
8-bit optimizer, matrix multiplication, and 8-bit and 4-bit quantization
functions. The library includes quantization primitives for 8-bit and
4-bit operations,
through bitsandbytes.nn.Linear8bitLt and bitsandbytes.nn.Linear4bit and
8-bit optimizers through bitsandbytes.optim module. These modules are
supported on AMD Instinct GPUs.

Installing bitsandbytes 
~~~~~~~~~~~~~~~~~~~~~~~

Install \`bitsandbytes\` for ROCm 6.0 (and later):

Run pip show bitsandbytes to show the information of the installed
bitsandbytes package. Tthe output should look like this:

Use bitsandbytes primitives
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use bitsandbytes with Transformers 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To load a Transformers model in 4-bit, set load_int_4bt=true in
BitsAndBytesConfig.

To load a model in 8-bit for inference, use the load_in_8bit option.

Model Acceleration libraries
============================

This section discusses model acceleration libraries.

Flash attention 2
-----------------

Flash Attention 2 is a technique that helps to reduce the number of
memory movements between GPU SRAM and HBM. By utilizing a tiling, Flash
Attention 2 improves the memory locality of a nested loop of the query,
key, and value of LLM’s Attention module, such as Multi-Head Attention
(MHA), Group-Query Attention (GQA), and Multi-Query Attention (MQA). The
reduced memory movements of Flash Attention 2 significantly reduce the
prefill (TTFT) latency for large batch size and long prompt sequence
length prompts.

|A computer screen shot of a diagram Description automatically
generated|

Installing Flash Attention 2 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ROCm software stack provides two different implementations of the Flash
Attention 2 modules, but they can be deployed interchangeably:

-  ROCm `Composable
   Kernel <https://github.com/ROCm/composable_kernel/tree/develop/example/01_gemm>`__
   (CK) Flash Attention 2

-  `OpenAI Triton <https://github.com/ROCm/triton>`__ Flash Attention 2

To install CK Flash Attention 2:

To install Triton Flash Attention 2 and run the benchmark:

Use Flash Attention 2 with Hugging Face Transformers and vLLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Hugging Face Transformers library can easily deploy the CK Flash
Attention 2 module by passing an argument
**”attn_implementation="flash_attention_2”** in the from_pretrained
class of Transformers’ library. This module significantly reduces model
inference and training latencies. The Triton Flash Attention 2 module
has been merged into the vLLM serving toolkit, which will be explained
in the following section.

Xformers
--------

Xformers is also known to improve the performance of attention modules.
Although Xformers attention performs very similarly to Flash Attention 2
due to its tiling behavior of query, key, and value, it’s widely used
for the LLMs and the Stable Diffusion models with the Hugging Face
Diffusers library.

Installing CK Xformers 
~~~~~~~~~~~~~~~~~~~~~~

To install CK Xformers:

Use the latest torch (2.3.0):

pip3 install torch torchvision torchaudio --index-url
https://download.pytorch.org/whl/rocm6.0

Use Xformers with vLLM
~~~~~~~~~~~~~~~~~~~~~~

Although Xformers are widely used for stable diffusion modules in the
Hugging Face Diffusers library, LLM also heavily utilizes this attention
module. The vLLM serving toolkit can utilize the Xformers attention
module instead of the CK and Triton Flash Attention 2 modules.

Pytorch built-in acceleration
=============================

`Pytorch compilation
mode <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>`__
synthesizes the model into a graph and then lowers it to prime
operators. These operators are compiled using TorchInductor, which uses
OpenAI Triton as a building block for GPU acceleration. One advantage of
PyTorch compilation mode is that its GPU kernels are written in Python,
making modifying and extending them easier. PyTorch compilation mode
often delivers higher performance, as model operations are fused before
runtime, which allows for easy deployment of high-performance kernels.

Pytorch compilation
-------------------

To utilize the Pytorch compilation mode, specific layers of the model
must be explicitly assigned as compilation targets. In the case of LLM,
where autoregressive token decoding generates dynamically changing
key/value sizes, limiting the key/value size to a static dimension,
**“max_cache_length"**, is necessary to utilize the performance benefits
of the Pytorch compilation.

Pytorch TunableOps
------------------

ROCm Pytorch (2.2.0 and later) allows users to use high-performance ROCm
GeMM kernel libraries through Pytorch's built-in TunableOps options.
This enables users to automatically pick up the best-performing GeMM
kernels from rocblas and hipblaslt libraries during runtime. During
warmup runs or offline profiling steps, users can create a GeMM Table
that enumerates the kernel information. During the model run, the
best-performing kernel substitutes **“torch.nn.functional.linear(input,
weight, bias=None)”** with the kernel specified in the GeMM table. The
`Tunable
Github <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/tunable/README.md>`__
page describes the options.

|A diagram of a software model Description automatically generated with
medium confidence|

LLM inference framework
=======================

This section describes the following LLM inference frameworks:

-  vLLM Inference

-  Text Generation Inference

vLLM Inference
--------------

vLLM from UC Berkely is famous for its paged attention algorithm that
can reduce memory consumption and increase throughput due to its paging
scheme. Instead of allocating GPU HBM memory for the maximum output
token lengths of the models, the paged attention of vLLM allocates GPU
HBM dynamically for its actual decoding lengths. This paged attention is
also effective when multiple requests share the same key and value
contents for a large value of beam search or multiple parallel requests.
vLLM also incorporates many recent LLM acceleration and quantization
algorithms, such as flash attention, cuda/hip graph, tensor parallel
multi-GPU, GPTQ, AWQ, and token speculation.

Installing vLLM
~~~~~~~~~~~~~~~

To install vLLM:

Using vLLM on a single GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~

To fully utilize accelerated modules in vLLM, install at least one of
the attention modules: CK Flash Attention 2, triton Flash Attention 2,
or Xformers. vLLM picks up one of the installed attention modules.
However, it’s advised to use triton Flash Attention by setting the
following environmental variable. **”VLLM_USE_FLASH_ATTN_TRITON=True”**

Using vLLM on multiple GPUs
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Text Generation Inference
-------------------------

Text Generation Inference (TGI) is LLM serving framework from Hugging
Face, and it also supports the majority of high-performance LLM
acceleration algorithms such as vLLM: flash attention, paged attention,
CUDA/HIP graph, tensor parallel multi-GPU, GPTQ, AWQ, and token
speculation. In addition to LLM serving capability, TGI also provides a
benchmark tool called **text-generation-benchmark**.

Installing Text Generation Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install TGI docker image:

Use Text Generation Inference on a single GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use Text Generation Inference on multiple GPUs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Kernel fusion using Composable Kernel (CK) Library 
==================================================

The Composable Kernel (CK) library provides a programming model for
writing performance-critical kernels for machine learning workloads. It
generates a general-purpose kernel during the compilation phase through
a C++ template, enabling developers to achieve operation fusions on
different data precisions.

This section outlines the CK build method, the high-level introduction
of one of CK instances, and the steps of constructing a C++ program
using the instance.

A detailed implementation of fused kernels for running SmoothQuant
quantized INT8 models on Instinct MI300X is discussed at the end of the
section.

Install Composable Kernel Library 
---------------------------------

1. | Clone CK source code from the Github repository and start the
     build:
   | git clone https://github.com/ROCm/composable_kernel.git
   | cd composable_kernel
   | mkdir build
   | cd build

2. | Configure and generate build files through cmake, the target device
     architecture GPU_TARGETS can be specified to speed up the building
     process:
   | cmake \\
   | -D CMAKE_PREFIX_PATH=/opt/rocm \\
   | -D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \\
   | -D CMAKE_BUILD_TYPE=Release \\
   | -D GPU_TARGETS="gfx942" \\
   | ..

3. | Build the entire CK library
   | make -j
   | The successful build looks like this:
   | …
   | [100%] Built target test_batched_gemm_softmax_gemm_permute_bf16
   | [100%] Built target test_batched_gemm_softmax_gemm_permute_fp16
   | [100%] Linking CXX executable
     ../../bin/test_batched_gemm_bias_softmax_gemm_permute_bf16
   | [100%] Built target
     test_batched_gemm_bias_softmax_gemm_permute_bf16

4. | Install CK. This will add CK header files, libraries, tools, and
     executable binaries into the system path of ROCm
   | make -j install

CK GEMM instance 
----------------

GEMM is a fundamental block in linear algebra, machine learning, and
deep neural networks. It’s defined as the operation:

E = alpha \* (A \* B) + beta \* D, with A and B as matrix inputs, alpha
and beta as scalar inputs, and D as a pre-existing matrix.

Take the commonly used linear transformation in a fully connected layer
as an example. These terms correspond to input activation (A), weight
(B), bias (D), and output (E), respectively. The example employs a
‘DeviceGemmMultipleD_Xdl_CShuffle’ struct as the fundamental instance to
explore the compute capability of AMD Instinct GPUs for the computation
of GEMM. The implementation of the kernel contains two phases:

1. Kernel template parameter definition

2. Templated kernel instantiation and running

The template parameters of the kernel are grouped into four parameter
types:

-  Parameters for determining matrix data precision

-  Parameters for determining matrix data layout

-  Parameters for determining extra operations on matrix elements

-  Performance-oriented Tunable parameters.

The template parameters of the selected GEMM kernel can be classified
into several groups.

As illustrated in the previous figure, these template parameter groups
must be defined properly before running the kernel.The following section
introduces the meaning of these parameters.

Matrix data precision
~~~~~~~~~~~~~~~~~~~~~

using ADataType = F16;

using BDataType = F16;

using AccDataType = F32;

using CShuffleDataType = F16;

using DDataType = F16;

using EDataType = F16;

ADataType and BDataType denote input matrix A and B data precision.
AccDataType determines the data precision used for representing the
multiply-add results of A and B elements. These results are stored in a
CShuffle module in local data share (LDS), which is a low-latency and
high-bandwidth explicitly addressed memory that is used for
synchronization within a workgroup LDS for later use, and
CShuffleDataType denotes the data precision of CShuffle in LDS.
DDataType denotes the data precision of the pre-existing D matrix stored
in GPU global memory, while EDatatype denotes the data precision of the
final output.

The CK kernel supports a fusion strategy so that CShuffle can be added
with a single pre-existing matrix in the same GPU kernel for better
performance. The above definitions demonstrate that A, B, D, and E are
half-precision floating-points. The multiply-add results of matrix A and
B are added with a pre-existing matrix D (half-precision), and the final
GEMM results are also half-precision floating-points.

Matrix data layout
~~~~~~~~~~~~~~~~~~

using ALayout = Row;

using BLayout = Col;

using DLayout = Row;

using ELayout = Row;

Following the convention of various linear algebra libraries, CK assumes
that the input matrix A is an M x K matrix, meaning the matrix has M
rows and K columns. Similarly, matrix B is assumed to be K x N, meaning
it has K rows and N columns. In computing, row-major order and
column-major order are commonly used ways to store matrices in linear
storage. After understanding the matrix storage pattern, the underlying
optimized memory access manner can be applied to achieve better
performance depending on the storage ordering of these matrices.

Matrix element operation 
~~~~~~~~~~~~~~~~~~~~~~~~

using AElementOp = PassThrough;

using BElementOp = PassThrough;

using CDEElementOp = AddRelu;

CK supports the pre-processing of the matrix prior to calculating GEMM,
that is, C = AElementOp(A) \* BElementOp(B), and post-processing of GEMM
results using the same way, that is, E = CEDElementOp(C, D). AElementOp
and BElementOp determine the operation applied to matrix A and B
separately prior to GEMM, which is achieved by binding the operation
with a C++ struct function. The above 'PassThrough' denotes no
operations are performed on the target matrix. CDEELementOp determines
the operations applied to CShuffle output and matrix D. The binding
function 'AddRelu' performs the addition of CShuffle output and matrix
D, and RELU operations to the addition result, it then passes the
results to matrix E.

struct AddRelu

{

\__host_\_ \__device_\_ void operator()(ck::half_t& e, const ck::half_t&
c, const ck::half_t& d) const

{

const ck::half_t x = c + d;

e = x > 0 ? x : 0;

}

};

Tunable Parameters 
~~~~~~~~~~~~~~~~~~

The CK kernel includes a series of tunable template parameters to
control the parallel granularity of the workload for the purpose of
achieving load balancing on different hardware platforms. These
parameters include Block Size, M/N/K Per Block, M/N per XDL, AK1, BK1,
and so on.

-  Block Size determines the number of threads in the thread block.

-  M/N/K Per Block determines the size of tile that each thread block is
   responsible for calculating

-  M/N Per XDL refers to M/N size for Instinct GPU Matrix Fused Multiply
   Add (MFMA) instructions operating on a per-wavefront basis.

-  A/B K1 is related to the data type. It can be any value ranging from
   1 to KPerBlock. To achieve the optimal load/store performance, 128bit
   per load is suggested. In addition, the A/B loading parameters must
   be changed accordingly to match the A/B K1 value; otherwise, it will
   result in compilation errors.

Due to the conditions for achieving computational load balancing on
different hardware platforms vary, optimal solution of tunable
parameters can be obtained through ckProfiler.

Kernel instantiation and running
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After determining the template parameters, the kernel is then
instantiated with actual arguments. For instance, you can either use
GetDeviceBuffer from CK’s custom struct DeviceMem to pass the element
values of the matrices that need to be calculated, or you can allocate
device buffer via hipMalloc, for which you need to ensure the device
buffer size can fit the matrix size. Users can also pass matrix elements
through the data_ptr method in the Tensor object if the matrix to be
calculated is of Tensor type.

The row and column, and stride information of input matrices are also
passed to the instantiation. For batch GEMM, additional batch count and
batch stride values must be passed in. The extra operations for pre and
post-processing are also passed with an actual argument. For example,
alpha and beta for GEMM scaling operations. Afterward, the instantiated
kernel is launched by the invoker, as illustrated below.

Templated kernel launching consists of kernel instantiation, making
arguments by passing in actual application parameters, creating an
invoker, and running the instance through the invoker.

Use CK 
------

This section uses the `CK GEMM
instance <https://github.com/ROCm/composable_kernel/tree/develop/example/01_gemm>`__
as a basic example, showing how to compile and link a program using the
instance. You can then organize any applications using this example.

Prepare source files
~~~~~~~~~~~~~~~~~~~~

In this example, CK GEMM instance is defined in **gemm_xdl_fp16.cpp**.
The data initialization and instance launch codes are implemented in
**run_gemm_example.inc**, The data structure definitions are included in
**common.hpp**. The CK utility library **libutility.a** is generated
during the installation.

Makefile common.hpp gemm_xdl_fp16.cpp libutility.a run_gemm_example.inc

Organize project compilation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Makefile used to automate the building procedure:

# Build target name

TARGET = main

# Target CK instance source

SOURCE = gemm_xdl_fp16.cpp

OBJ = $(SOURCE).o

# Specify the target device where CK instance binary run

ARCH = gfx942

# Specify hipcc path

CXX = /opt/rocm/bin/hipcc

# HIP_PLATFORM_AMD is defined if the HIP platform targets AMD

CXX_DEFINES = -DCK_ENABLE_FP16 -D__HIP_PLATFORM_AMD__=1

# Other compilation options

CXX_FLAGS = -O3 -DNDEBUG -std=c++17 --offload-arch=$(ARCH)

# This library is generated by CK

LIB = libutility.a

$(TARGET): $(OBJ)

$(CXX) $(OBJ) -o $(TARGET) $(LIB)

$(OBJ): $(SOURCE)

$(CXX) $(CXX_DEFINES) $(CXX_FLAGS) -o $(OBJ) -c $(SOURCE)

Build the project 
~~~~~~~~~~~~~~~~~

# Build the project

make

# Generate the executable binary, that is, main

Makefile common.hpp gemm_xdl_fp16.cpp gemm_xdl_fp16.cpp.o libutility.a
main run_gemm_example.inc

Run the executable binary 
~~~~~~~~~~~~~~~~~~~~~~~~~

#arg1: verification (0=no, 1=yes)

#arg2: initialization (0=no init, 1=integer value, 2=decimal value)

#arg3: run kernel # of times (>1)

./main 0 1 5

Once completed, it should show the following output:

a_m_k: dim 2, lengths {3840, 4096}, strides {4096, 1}

b_k_n: dim 2, lengths {4096, 4096}, strides {4096, 1}

c_m_n: dim 2, lengths {3840, 4096}, strides {4096, 1}

Perf: 1.07575 ms, 119.776 TFlops, 89.6758 GB/s,
DeviceGemm_Xdl_CShuffle<Default, 256, 256, 128, 32, 8, 2, 32, 32, 4, 2,
8, 4, 1, 2> LoopScheduler: Interwave, PipelineVersion: v1

Develop Fused INT8 Kernels for SmoothQuant Models 
-------------------------------------------------

`SmoothQuant <https://github.com/mit-han-lab/smoothquant>`__ (SQ) is a
quantization algorithm that enables an INT8 quantization of both weights
and activations for all the matrix multiplications in LLM. The required
GPU kernel functionalities used to accelerate the inference of SQ models
on Instinct GPUs are shown in the following table.

Functionalities used to implement SmoothQuant model inference

+----------------------------------+----------------------------------+
| Functionality Descriptions       | Corresponding wrappers           |
+==================================+==================================+
| E = alpha×A×B + beta×D, where A, | E = Linear_ABDE_I8(A, B, D,      |
| B, D, E are INT8 2-D tensors;    | alpha, beta)                     |
+----------------------------------+----------------------------------+
| E = ReLU(alpha×A×B + beta×D),    | E = Linear_ReLU_ABDE_I8(A, B, D, |
| where A, B, D, E are INT8 2-D    | alpha, beta)                     |
| tensors;                         |                                  |
+----------------------------------+----------------------------------+
| E = alpha×A×B + beta×D, where A, | E = Linear_AB_I8_DE_F32(A, B, D, |
| B are INT8 2-D tensors, D and E  | alpha, beta)                     |
| are FP32 2-D tensors;            |                                  |
+----------------------------------+----------------------------------+
| E = alpha×A×B, where A, B, E are | E = BMM_ABE_I8(A, B, alpha)      |
| INT8 3-D tensors,                |                                  |
|                                  |                                  |
| for example, batched matrices    |                                  |
+----------------------------------+----------------------------------+
| E = alpha×A×B, where A, B are    | E = BMM_AB_I8_E_F32(A, B, alpha) |
| INT8 3-D tensors, E is FP32 3-D  |                                  |
| tensor,                          |                                  |
|                                  |                                  |
| for example, batched matrices    |                                  |
+----------------------------------+----------------------------------+

Operation flow analysis
~~~~~~~~~~~~~~~~~~~~~~~

The following section discusses the analysis of the operation flow of
Linear_ReLU_ABDE_I8. The rest wrappers in Table 4.1 can be analyzed
similarly. The first operation in the process is to perform the
multiplication of input matrices A and B, the resulting matrix C is then
scaled with alpha to obtain T1.

At the same time, the process performs a scaling operation on D elements
to obtain T2. Afterward, the process performs the operations of matrix
addition between T1 and T2, element activation calculation using ReLU,
and element rounding sequentially. The operations to generate E1, E2,
and E are encapsulated and completed by a user-defined template function
in CK (given in the next sub-section). This template function is
integrated into the fundamental instance directly during the compilation
phase so that all these steps can be fused in a single GPU kernel.

Select a fundamental CK instance 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CK library contains many fundamental instances that implement different
functions, you need to rely on your experience to identify the names of
numerous CK instances and determine whether they meet the target
functional requirements at the beginning.

Secondly, it is necessary to consider whether the format of input data
meets the actual calculation needs. For SQ models, 8-bit integer data
format (INT8) is applied for matrix calculations.

Thirdly, users need to consider the platform for implementing CK
instances. The instances with ‘xdl’ suffix can only run on AMD Instinct
GPUs after being compiled and cannot run on Radeon series GPUs. This is
due to the underlying instruction set for implementing these basic
instances using device-related instruction sets.

For this design case,
`DeviceBatchedGemmMultiD_Xdl <https://github.com/ROCm/composable_kernel/tree/develop/example/24_batched_gemm>`__
can be employed as the fundamental instance to implement the five
functionalities in the Table.

Develop the complete function 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since the inference of SQ quantized models relies on using PyTorch and
Transformer libraries and a tensor type is used in torch for
representing matrices and vectors, the C++ data types in CK should be
replaced by **torch::tensor type**.

The data types of the input and output matrices should be a tensor type.
In GEMM, the A and B inputs are two-dimensional matrices, and the
required input matrices of the selected fundamental CK instance are
three-dimensional matrices. Therefore it’s necessary to convert the
input 2-D tensors to 3-D tensors by using tensor's **unsqueeze()**
method before passing these matrices to the instance. For batched GEMM
in Table 4.1, ignore this step.

Obtain the M, N, and K values using input tensor size values. This
stride size information is used to reshape the input vector D and
allocate the storage space of tensor E. In addition, stride reflects the
exact size of continuous elements in memory, which are passed as
important parameters to the fundamental instance for GPU kernel use.

ADataType, BDataType and D0DataType are used to denote the data
precision of the input tensor A, B and D, respectively. EDataType is
used to denote the data precision of output tensor E.

These parameters are specified to I8 data format (8-bit integer data
format) to meet the kernel design requirements. AccDataType determines
the data precision used for representing the multiply-add results of A
and B elements. Generally, a larger range data type is applied to store
the multiply-add results of A and B to avoid result overflow. That is,
I32 is applied in this case. The CShuffleDataType I32 data type
indicates that the multiply-add results continue to be stored in LDS as
a I32 data format.

All of the above is implemented through the following code.

Following the convention of various linear algebra libraries, row-major
and column-major orders are used to denote the ways of storing matrices
in linear storage. The advantage of specifying matrix B as column major
is that all the relevant matrix elements are stored continuously in GPU
global memory when a row in A is multiplied by a column in B, which can
help GPU achieve data consistency access and improve access performance.

In CK, 'PassThrough' is a struct denoting if an operation is applied to
the tensor it binds to.

To fuse the operations between E1, E2, and E introduced in section 4.1,
a custom C++ struct, '**ScaleScaleAddRelu**', is defined and bound to
CDEELementOp. It determines the operations that will be applied to
CShuffle (A×B results), tensor D, alpha, and beta.

In the binding struct, the operator() performs an addition operation
between CShuffle and matrix D, RELU operation on addition results, and
the rounding operation of output elements, and then returns the results
to E.

The original input tensors need to be padded to meet GPU tile-based
parallelism.

The template parameters of the target fundamental instance are
initialized with the above parameters. In addition, the template
includes default tunable parameters. For specific tuning methods, please
refer to the previous content.

Return the address of the first element of tensors:

The fundamental instance is then initialized and run with actual
arguments:

The output of the fundamental instance is a calculated batched matrix E
(batch, M, N). Before the return, it needs to be converted to a 2-D
matrix.

Bind to Python 
~~~~~~~~~~~~~~

Once these functions are written in C++ and torch::Tensor, you can use
pybingd11 to bind the functions into Python. For the example, the
necessary binding code for exposing the functions in Table 4.1 spans
with few lines.

Build the C++ extension by writing a **setup.py** script that uses
setuptools to compile the C++ code. A reference writing of the
**setup.py** script is as follows.

Run python **setup.py** install to build and install the extension. It
should look something like this:

|image2|

INT8 model performance 
~~~~~~~~~~~~~~~~~~~~~~

The implementation architecture of running Smoothquant models on AMD
MI300X GPUs

For the target `SQ quantized
model <https://huggingface.co/mit-han-lab/opt-13b-smoothquant>`__, each
decoder layer contains three major components, which are attention
calculation, layer normalization, and linear transformation in fully
connected layers. The corresponding implementation classes for these
components are:

-  Int8OPTAttention

-  W8A8B8O8LinearReLU

-  W8A8BF32OF32Linear

Note that the implementation of LayerNormQ module is implemented by
torch native module for the example. These classes' underlying
implementation logits will harness the Table functions. The figure below
illustrates the advantages of running inference on the INT8 models over
the original FP16 models on MI300X.

|image3|

The GPU memory footprint comparisons (a) and model inference speed
comparisons (b) between the original FP16 models and the Smoothquant
INT8 models on an AMD MI300X GPU.

Model profiling and debugging
=============================

Pytorch built-in profiler
-------------------------

Pytorch profiler can be invoked inside the Python script, enabling users
to collect information while the script is running. By using Pytorch
profiler, users can record CPU and GPU-related performance metrics.
These numbers can be viewed by an open-source profile visualization
tool, such as Perfetto UI.

-  `Pytorch profiler <Pytorch%20profiler>`__

-  `Pytorch profiler tutorial <Pytorch%20profiler%20tutorial>`__

-  `Perfetto <Perfetto>`__

Profile A

-  Unopt: vanilla transformers

Profile B

-  Oopt: vllm + FA + PA + TP + cuda_graph

|A screenshot of a computer Description automatically generated|

|A screenshot of a computer Description automatically generated|

Rocprof, Omniperf, Thread trace viwer (TTV)
-------------------------------------------

`Rocprof <https://rocmdocs.amd.com/projects/rocprofiler/en/latest/rocprofv1.html>`__
is our profiling tool to collect various performance metrics of a kernel
execution.

`Omniperf <https://github.com/ROCm/omniperf>`__ builds upon rocprof but
provides more guided analysis.

Thread trace viewer (TTV) can visualize the thread trace captured
through Rocprof. It shows the execution and halt for each thread so that
you can pinpoint the hotspot. More to come.

We will provide a docker image that has omniperf and thread trace viewer
preinstalled. Also tutorials on how to use them.

Model debugging
~~~~~~~~~~~~~~~

Some examples of showing how to find the root cause if running an LLM
model produce abort/error or wrong result or poor performance due to
using wrong kernels or some kernel not being used …

UNDER CONSTRUCTION

Debugging Memory Access Faults
------------------------------

Identifying the faulting kernel is often enough to triage a memory
access fault. To that end, the ROCm Debug Agent can trap a memory access
fault and provide a dump of all active wavefronts that caused the error
as well as the name of the kernel. The `AMD ROCm Debug Agent Library
README <https://github.com/ROCm/rocr_debug_agent>`__ provides full
instructions, but to summarize:

-  Compiling with -ggdb -O0 is recommended but not required.

-  HSA_TOOLS_LIB=/opt/rocm/lib/librocm-debug-agent.so.2
   HSA_ENABLE_DEBUG=1 ./my_program

When the debug agent traps the fault, it will produce an extremely
verbose output of all wavefront registers and memory content.
Importantly, it also prints something like:

Disassembly for function vector_add_assert_trap(int*, int*, int*):

code object:
file:////rocm-debug-agent/build/test/rocm-debug-agent-test#offset=14309&size=31336

loaded at: [0x7fd4f100c000-0x7fd4f100e070]

The kernel name and the code object file should be listed. In the
example above, the kernel name is vector_add_assert_trap, but this might
also look like

Disassembly for function
memory:///path/to/codeobject#offset=1234&size=567:

In this case, it is an in-memory kernel that was generated at runtime.
Using the env var

ROCM_DEBUG_AGENT_OPTIONS="--all --save-code-objects"

the debug agent will save all code objects to the current directory (use
--save-code-objects=[DIR] to place them in another location). The code
objects will be renamed from the URI format with special characters
replaced by ‘_’. Use llvm-objdump to disassemble the indicated in-memory
code object that has now been saved to disk. The name of the kernel is
often found inside the disassembled code object.

llvm-objdump --disassemble-all path/to/code-object.co

Consider turning off memory caching strategies both within the ROCm
stack and PyTorch, where possible. This will give the debug agent the
best chance at finding the memory fault where it originates. Otherwise,
it could be masked by writing past the end of a cached block within a
larger allocation.

PYTORCH_NO_HIP_MEMORY_CACHING=1

HSA_DISABLE_FRAGMENT_ALLOCATOR=1

AMD Triton Kernel Performance Optimization
==========================================

This section introduces the general steps for Triton kernel
optimization. Overall, Triton kernel optimization is similar to CUDA/HIP
kernel optimization. This section describes the aspects.

Hardware resource utilization
-----------------------------

Each GPU has many Compute Units (CUs), and different CUs do computation
in parallel, so the first thing to consider is how many CUs a kernel can
allocate its task to. For AMD MI300X, the grid should have a least 1024
thread blocks/workgroups (WGs). To increase hardware utilization, more
parallelism needs to be found in the algorithm (for example, using
larger split-K for GEMMs).

Hardware resources can be queried with the command \`rocminfo\` (in the
folder \`/opt/rocm/bin`). For instance, wan query # of computes, # of
SIMD, and wavefront size as:

rocminfo \| grep "Compute Unit"

rocminfo \| grep "SIMD"

rocminfo \| grep "Wavefront Size"

For AMD MI300X, there are 304 CUs, 4 SIMD per CU, and the wavefront size
(warp size) is 64.

Autotunable kernel configurations and environment variables
-----------------------------------------------------------

This is about the amount of memory access and computation assigned to
each CU. It is related to the usage of LDS, register and the scheduling
of different tasks on a CU.

The following kernel arguments can be tuned.

num_stages=n
~~~~~~~~~~~~

On AMD GPUs, set num_stages according to the following rules:

-  For kernels with single GEMM, set to 0

-  For kernels with two GEMMs fused (FlashAttention, or any other kernel
   that fuses 2 GEMMs), set to 1

-  For kernels that fuse a single GEMM with another non GEMM operator
   (for example reLU activation), set to 0

-  For kernels that have no GEMMs, set to 1

waves_per_eu=n
~~~~~~~~~~~~~~

See `Understand/Compute the occupancy of the
kernel <#understandcompute-the-occupancy-of-the-kernel>`__ for more
information about how to compute occupancy. It hints to the compiler to
reduce VGPR so that occupancy = n could be achieved. This only helps if
both of the following satisfy:

-  The occupancy of the kernel is limited by VGPR usage.

-  The current VGPR usage is only a few above a boundary in table 1.

For example, according to the table, the available VGPR is 512 per
Execution Unit (EU), and VGPU is allocated at the unit of 16. If the
current VGPR usage is 170, the actual requested VGPR will be 176, so the
occupancy is only 2 waves/CU since 176 x 3 > 512. Then if you set
waves_per_eu to 3, the LLVM backend tries to bring VGPR usage down so
that it might fit 3 waves/EU.

BLOCK_M, BLOCK_N, BLOCK_K
~~~~~~~~~~~~~~~~~~~~~~~~~

Tile sizes need to be tuned. You want tile sizes large enough to
maximize the efficiency of memory-to-computation ratio, but small enough
to parallelize the greatest number of WGs at the grid level.

matrix_instr_nonkdim
~~~~~~~~~~~~~~~~~~~~

This is an experimental feature for FA-like kernels. It can choose the
size of MFMA instruction used. For GEMM kernels on AMD MI300X,
mfma_16x16 performs better than mfma_32x32, even for large tile/GEMM
sizes.

-  Matrix_instr_nonkdim = 16: mfma_16x16 is used

-  Matrix_instr_nonkdim = 32: mfma_32x32 is used

OPTIMIZE_EPILOGUE 
~~~~~~~~~~~~~~~~~

This is an environment variable that should be turned on (set to 1) in
most cases. It removes the convert_layout in the epilogue. By default,
the results of MFMA instruction are converted to blocked layout, which
leads to global_store with maximum vector length, that is
global_store_dwordx4.

This is done implicitly with LDS as the intermediate buffer to achieve
data exchange between threads. Padding is used in LDS to avoid bank
conflicts. This usually leads to extra LDS usage, which might reduce
occupancy. Setting OPTIMIZE_EPILOGUE=1 will have the effect of storing
the result in the MFMA layout. This reduces the efficiency of global
stores but has an insignificant influence on kernel execution time.

Note that this variable is not turned on by default because it only
works with tt.store but not tt.atomic_add, which is used in split-k and
stream-k GEMM kernels. In the future, it might be enabled with
tt.atomic_add and turned on by default.

Memory access efficiency
------------------------

The GPU has global memory, local data share (LDS, shared memory), and
register. The memory has high access latency and is large. LDS access
has much lower latency but is small. Register access is the fastest yet
smallest among the three.

Overall, the data in global memory should be loaded and stored as few
times as possible. If different threads in a block need to access the
same data, these data should be first transferred from global memory to
LDS, then accessed by different threads in a workgroup.

IR analysis
-----------

In Triton, there are several layouts including blocked, shared, sliced,
and MFMA.

From the Triton GPU IR, you can know in which memory each computation is
performed. The following is a snippet of IR from the Flash Attention
(FA) decode int4 KV program. It is to dequantize the int4 KV from int4
data type to fp16.

%190 = tt.load %189 {cache = 1 : i32, evict = 1 : i32, isVolatile =
false} : tensor<1x64xi32, #blocked6> loc(#loc159)

%266 = arith.andi %190, %cst_28 : tensor<1x64xi32, #blocked6>
loc(#loc250)

%267 = arith.trunci %266 : tensor<1x64xi32, #blocked6> to
tensor<1x64xi16, #blocked6> loc(#loc251)

%268 = tt.bitcast %267 : tensor<1x64xi16, #blocked6> -> tensor<1x64xf16,
#blocked6> loc(#loc252)

%269 = triton_gpu.convert_layout %268 : (tensor<1x64xf16, #blocked6>) ->
tensor<1x64xf16, #shared1> loc(#loc252)

%270 = tt.trans %269 : (tensor<1x64xf16, #shared1>) -> tensor<64x1xf16,
#shared2> loc(#loc194)

%276 = triton_gpu.convert_layout %270 : (tensor<64x1xf16, #shared2>) ->
tensor<64x1xf16, #blocked5> loc(#loc254)

%293 = arith.mulf %276, %cst_30 : tensor<64x1xf16, #blocked5>
loc(#loc254)

%295 = arith.mulf %292, %294 : tensor<64x32xf16, #blocked5> loc(#loc264)

%297 = arith.addf %295, %296 : tensor<64x32xf16, #blocked5> loc(#loc255)

%298 = triton_gpu.convert_layout %297 : (tensor<64x32xf16, #blocked5>)
-> tensor<64x32xf16, #shared1> loc(#loc255)

%299 = tt.trans %298 : (tensor<64x32xf16, #shared1>) ->
tensor<32x64xf16, #shared2> loc(#loc196)

%300 = triton_gpu.convert_layout %299 : (tensor<32x64xf16, #shared2>) ->
tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth
= 4}>> loc(#loc197)

From the IR, you can see i32 data is loaded from global memory to
registers. With a few element-wise operations in registers, then it is
stored in shared memory for the transpose operation, which needs data
movement across different threads. With transpose done, it is loaded
from LDS to register again, and with a few more element-wise operations,
they are stored in LDS again. Last step is loaded from LDS to registers
and converted to the dot operand layout.

From the IR, you can see that it uses the LDS twice; one is for the
transpose, and the other is to convert the blocked layout to a dot
operand layout.]

Assembly analysis
-----------------

In the ISA, ensure **global_load_dwordx4** is used, especially when the
load happens in the loop.

In most cases, the LDS load and store should use \_b128 as well to
minimize the number of LDS access instructions. Note that upstream
(Phantom dubs this as backend) might not have \_b128 LDS read/write, so
it uses \_b64. For most cases, no matter if you use fork or upstream,
the LDS access should have \_b64 vector width.

The AMD ISA has **s_waitcnt** instruction to synchronize the dependency
of memory access and computations. The **s_waitcnt** instructions can
have two signals, typically in the Triton context

-  **lgkmcnt(n):** lgkm stands for LDS, GDS, Constant and Message. For
   our context, it is often related to LDS access. The number n here
   means the number of such accesses can be left out to continue. For
   example, 0 means all lgkm access must finish before continuing, and 1
   means only 1 lgkm access can be still running asynchronously before
   proceeding.

-  **vmcnt(n):** vm means vector memory. This happens when vector memory
   is accessed, for example, when global load moves from global memory
   to vector memory. The variable n means the same thing as the above.

The general guidelines are:

-  Vectorize memory access as much as possible.

-  Ensure synchronization is done efficiently.

-  Overlap of instructions to hide latency, but it requires thoughtful
   analysis of the algorithms.

-  If you find inefficiencies, you can trace it back to LLVM IR, TTGIR
   and even TTIR to see where the problem comes from. If you find it
   during compiler optimization, activate the MLIR dump and check which
   optimization pass caused the problem.

Understand/Compute the occupancy of the kernel
----------------------------------------------

1. Get the VGPR count, search for .vgpr_count in the ISA. For example,
N.

2. Get the allocated LDS following the steps. For example, L for the
kernel.

   a. export MLIR_ENABLE_DUMP=1

   b. rm -rf ~/.triton/cache

   c. python kernel.py \| \| grep "triton_gpu.shared = " \| tail -n 1

   d. Look for something like **triton_gpu.shared = 65536**. It means
   65536 bytes LDS is allocated for the kernel.

3. Get number of waves per workgroup following the steps (say you got
nW)

   a. export MLIR_ENABLE_DUMP=1

   b. rm -rf ~/.triton/cache

   c. python kernel.py \| \| grep "triton_gpu.num-warps " \| tail -n 1

   d. Look for something like “triton_gpu.num-warps" = 8 it means 8
   waves per workgroup

4. Compute occupancy limited by VGPR based on N according to table 1 in
this link. For example, waves per EU as occ_vgpr.

5. Compute occupancy limited by LDS based on L by: occ_lds = floor(65536
/ L).

6. Then the occupancy is occ = min(floor(occ_vgpr \* 4 / nW), occ_lds)
\* nW / 4

   a. occ_vgpr \* 4 gives the total number of waves on all 4 EUs (SIMDs)
   per CU

   b. floor(occ_vgpr \* 4 / nW) gives the occupancy of workgroups per CU
   regrading VGPR usage

   c. The true occ is the minimum of the two.

PyTorch inductor Triton tuning knobs
------------------------------------

To enable gemm/conv lowerings to Triton, it requires use of inductor’s
max_autotune mode. This benchmarks a static list of triton configs (conv
configs for max autotune + matmul configs for max autotune) and uses the
fastest for each shape. Note that the Triton is not used if regular
MIOpen/rocBlas is faster for a specific operation.

\`torch._inductor.config.max_autotune = True\` or
TORCHINDUCTOR_MAX_AUTOTUNE=1

Or for more fine-grained control:

\`torch._inductor.config.max_autotune.pointwise = True\` - to enable
tuning for pointwise/reduction ops

\`torch._inductor.config.max_autotune_gemm = True\` - to enable
tuning/lowering of mm/convs

\`torch._inductor.max_autotune_gemm_backends/TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS\`
- to select the candidate backends for mm autotuning Defaults to
“TRITON,ATEN”, NV also includes CUTLASS tuning option. Limiting this to
“TRITON” might improve performance by enabling more fused mm kernels
instead of going to rocBlas

For **mm tuning coordinate_descent** tuning might improve performance,
which attempts

‘torch._inductor.config.coordinate_descent_tuning =
True`/TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1

Inference can see large improvements on AMD GPUs by utilizing
\`torch._inductor.config.freezing=True`/TORCHINDUCTOR_FREEZING=1, which
inlines weights as constants and enables constant folding optimizations.

Enabling inductor’s cpp_wrapper might improve overhead. This generates
C++ code which launches Triton binaries directly with
hipModuleLaunchKernel and relies on hipification.

For NHWC convolutions workloads
\`torch._inductor.config.layout_optimization=True`/TORCHINDUCTOR_LAYOUT_OPTIMIZATION=\`
can help be enforcing channels_last format throughout the graph avoiding
any additional transposes added by inductor. Note that
PYTORCH_MIOPEN_SUGGEST_NHWC=1 is recommended if using this.

Extracting the Triton kernel TORCH_COMPILE_DEBUG creates a
torch_compile_debug/ directory at current path, in the output_code.py
the code-strings for the triton kernels that are defined. Manual work is
then required to strip out the kernel and create kernel
compilation/launch via Triton.

For advanced matmul/conv config tuning, the inductor-gemm-tuner can
help. This implements the triton conv/mm implementations used upstream
and allows specification of inputs and config tuning search space if new
tunings are found can be added to the autotune list.

Miscellaneous
-------------

Performance-critical HIP provides an environment variable, “export
HIP_FORCE_DEV_KERNARG=1,” that can put HIP kernel arguments directly to
device memory to reduce the latency of accessing kernel arguments. It
can reduce 2 to 3 us for some kernels. Setting this variable for the FA
decode containing splitK and reduced kernels can reduce the total time
by ~6us in the benchmark test.

Set the clock for deterministic. Use the command \`rocm-smi
--setperfdeterminism 1900\` to set the max clock speed to 1900MHz
instead of the default 2100MHz. This can reduce the chance of clock
speed decrease due to chip high temperature by setting a lower cap. This
setting can be restored to default with \`rocm-smi -r`.

Set numa autobalance. Run the command \`cat
/proc/sys/kernel/numa_balancing\` to check the current settings. Output
0 indicates this setting is available. If output is 1, run the command
\`sudo sh -c \\'echo 0 > /proc/sys/kernel/numa_balancing\` to set this.

For these settings, we created a script to do ‘set’, ‘reset’, ‘checking’
of the above environments. The script is located at
`env_check.sh <https://amdcloud-my.sharepoint.com/:u:/g/personal/shxiao_amd_com/EZj5Sg0av7NBiprBwS8HKCEBHuBYnOHoCB2lFUoGKHH1Gg?e=twfqAU>`__

#!/bin/bash

function print_usage {

echo " Usage: env_set.sh set/reset/check"

echo " set: configure the settings in this script"

echo " reset: reset to default settings"

echo " check: check the current settings"

}

function set_env {

export HIP_FORCE_DEV_KERNARG=1

rocm-smi --setperfdeterminism 1900

sudo sh -c echo 0 > /proc/sys/kernel/numa_balancing

}

function reset_env {

unset HIP_FORCE_DEV_KERNARG

rocm-smi -r

sudo sh -c echo 1 > /proc/sys/kernel/numa_balancing

}

function check_env {

echo ""

echo "---------------------------------------------------------------"

echo ""

# check the flag to force kernel to be on device memory

echo "1. Check forcing kernel args on device memory"

dev_kernarg=$(env \| grep HIP_FORCE_DEV_KERNARG)

if [ -z $dev_kernarg ]

then

echo " no setting for forcing kernel args on device memory"

echo " run the command \\"export HIP_FORCE_DEV_KERNARG=1\" to force it"

else

echo " env var \\"HIP_FORCE_DEV_KERNARG\" for forcing kernel args on
device"

echo " memory is set, we have HIP_FORCE_DEV_KERNARG="
$HIP_FORCE_DEV_KERNARG

if [ "$HIP_FORCE_DEV_KERNARG" -eq 0 ]

then

echo " env var HIP_FORCE_DEV_KERNARG is 0, set it to 1 by:"

echo " command \\"export HIP_FORCE_DEV_KERNARG=1\""

fi

fi

echo ""

echo ""

echo "2. Set perfdeterminism, highest frequency"

echo " run the command \\"rocm-smi -a \| grep sclk\" to check highest
frequency."

echo " you can run the command \\"rocm-smi --setperfdeterminism # (for
example 1900)\" to"

echo " set clock frequency limit to get minimal performance, which is
more reproducible"

echo " you can restore the setting by running \\"rocm-smi
--resetperfdeterminism\""

echo ""

echo ""

echo "3. Check numa autobalance"

autobal=$(cat /proc/sys/kernel/numa_balancing)

if [ $autobal -ne 0 ]

then

echo " run the command \\"sudo sh -c \\'echo 0 >
/proc/sys/kernel/numa_balancing\'\""

echo " to set numa autobalance".

echo " you can disable it with \\"sudo sh -c \\'echo 1 >
/proc/sys/kernel/numa_balancing\'\""

else

echo " numa autobalance is checked with:"

echo " (cat /proc/sys/kernel/numa_balancing)=0"

fi

echo ""

echo "---------------------------------------------------------------"

echo ""

}

if [ $# -eq 0 ]

then

echo " \\"env_set.sh -h\" for help info"

print_usage

exit 1

fi

input=$1

if [ $1 == "set" ]

then

set_env

elif [ $1 == "reset" ]

then

reset_env

elif [ $1 == "check" ]

then

check_env

else

print_usage

fi

Tunable Op has been merged into Pytorch. The behavior of TunableOp is
easily manipulated through environment variables, though you could use
the C++ interface of at::cuda::tunable::getTuningContext(). A Python
interface to the TuningContext does not yet exist.

The default is 0, which means only 1 iteration is attempted.

There’s an overhead to tuning. To try and minimize the overhead, only a
limited number of iterations of a given operation are attempted. If you
set this to 10, each solution for a given operation can run as many
iterations as possible within 10ms. There is a hard-coded upper limit of
100 iterations attempted per solution. This is a tuning parameter; if
you want the tunings to be chosen based on an average over multiple
iterations, increase the allowed tuning duration.

.. |A computer screen shot of a diagram Description automatically generated| image:: media/image1.png
   :width: 4.68657in
   :height: 3.33242in
.. |A diagram of a software model Description automatically generated with medium confidence| image:: media/image2.png
   :width: 4.95893in
   :height: 3.76398in
.. |image2| image:: media/image3.png
   :width: 5.85066in
   :height: 2.33093in
.. |image3| image:: media/image4.png
   :width: 2.92014in
   :height: 1.77917in
.. |A screenshot of a computer Description automatically generated| image:: media/image6.png
   :width: 6.26806in
   :height: 1.26319in
.. |A screenshot of a computer Description automatically generated| image:: media/image7.png
   :width: 6.26806in
   :height: 1.34653in

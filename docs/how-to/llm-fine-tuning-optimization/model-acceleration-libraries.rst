.. meta::
   :description: How to fine-tune LLMs with ROCm
   :keywords: ROCm, LLM, fine-tuning, usage, tutorial, Flash Attention, Hugging Face, xFormers, vLLM, PyTorch

****************************
Model acceleration libraries
****************************

This section discusses model acceleration techniques and libraries to improve memory efficiency and performance.

.. _acceleration-flash-attention:

Flash Attention 2
=================

Flash Attention is a technique designed to reduce memory movements between GPU SRAM and high-bandwidth memory (HBM). By
using a tiling approach, Flash Attention 2 improves memory locality in the nested loops of query, key, and value
computations within the Attention modules of LLMs. These modules include Multi-Head Attention (MHA), Group-Query
Attention (GQA), and Multi-Query Attention (MQA). This reduction in memory movements significantly decreases the
time-to-first-token (TTFT) latency for large batch sizes and long prompt sequences, thereby enhancing overall
performance.

.. image:: ../../data/how-to/llm-fine-tuning-optimization/attention-module.png
   :alt: Attention module of a large language module utilizing tiling
   :align: center

Installing Flash Attention 2 
----------------------------

ROCm provides two different implementations of Flash Attention 2 modules. They can be deployed interchangeably:

*  ROCm `Composable Kernel <https://github.com/ROCm/composable_kernel/tree/develop/example/01_gemm>`_
   (CK) Flash Attention 2

*  `OpenAI Triton <https://triton-lang.org/main/index.html>`_ Flash Attention 2

.. tab-set::

   .. tab-item:: CK Flash Attention 2

      To install CK Flash Attention 2, use the following commands.

      .. code-block:: shell

         # Install from source
         git clone https://github.com/ROCm/flash-attention.git
         cd flash-attention/
         GPU_ARCHS=gfx942 python setup.py install #MI300 series

      Hugging Face Transformers can easily deploy the CK Flash Attention 2 module by passing an argument
      ``attn_implementation="flash_attention_2"`` in the ``from_pretrained`` class.

      .. code-block:: python

         import torch
         from transformers import AutoModelForCausalLM, AutoTokenizer
         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
         model_name = "NousResearch/Meta-Llama-3-8B"

         tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.float16, use_fast=False)
         inputs = tokenizer('Today is', return_tensors='pt').to(device)

         model_eager = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, attn_implementation="eager").cuda(device)
         model_ckFAv2 = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, attn_implementation="flash_attention_2").cuda(device)

         print("eager GQA: ", tokenizer.decode(model_eager.generate(**inputs, max_new_tokens=10)[0], skip_special_tokens=True))
         print("ckFAv2 GQA: ", tokenizer.decode(model_ckFAv2.generate(**inputs, max_new_tokens=10)[0], skip_special_tokens=True))

         #  eager GQA:  Today is the day of the Lord, and we are the
         # ckFAv2 GQA: Today is the day of the Lord, and we are the

   .. tab-item:: Triton Flash Attention 2

      The Triton Flash Attention 2 module is implemented in Python and uses OpenAI’s JIT compiler. This module has been
      upstreamed into the vLLM serving toolkit, discussed in :doc:'llm-inference-frameworks'. 

      1. To install Triton Flash Attention 2 and run the benchmark, use the following commands.

         .. code-block:: shell

            # Install from the source
            pip uninstall pytorch-triton-rocm triton -y
            git clone https://github.com/ROCm/triton.git 
            cd triton/python
            GPU_ARCHS=gfx942 python setup.py install #MI300 series
            pip install matplotlib pandas

      2. To test, run the Triton Flash Attention 2 performance benchmark.

         .. code-block:: shell
         
            # Test the triton FA v2 kernel
            python https://github.com/ROCm/triton/blob/triton-mlir/python/perf-kernels/flash-attention.py
            # Results (Okay to release TFLOPS number ???)
            fused-attention-fwd-d128:
                BATCH    HQ    HK  N_CTX_Q  N_CTX_K      TFLOPS
            0    16.0  16.0  16.0   1024.0   1024.0  287.528411
            1     8.0  16.0  16.0   2048.0   2048.0  287.490806
            2     4.0  16.0  16.0   4096.0   4096.0  345.966031
            3     2.0  16.0  16.0   8192.0   8192.0  361.369510
            4     1.0  16.0  16.0  16384.0  16384.0  356.873720
            5     2.0  48.0  48.0   1024.0   1024.0  216.916235
            6     2.0  48.0  48.0   2048.0   1024.0  271.027578
            7     2.0  48.0  48.0   4096.0   8192.0  337.367372
            8     2.0  48.0  48.0   8192.0   4096.0  363.481649
            9     2.0  48.0  48.0  16384.0   8192.0  375.013622
            10    8.0  16.0  16.0   1989.0  15344.0  321.791333
            11    4.0  16.0  16.0   4097.0    163.0  122.104888
            12    2.0  16.0  16.0   8122.0   2159.0  337.060283
            13    1.0  16.0  16.0  16281.0      7.0    5.234012
            14    2.0  48.0  48.0   1021.0   1020.0  214.657425
            15    2.0  48.0  48.0   2001.0   2048.0  314.429118
            16    2.0  48.0  48.0   3996.0   9639.0  330.411368
            17    2.0  48.0  48.0   8181.0   1021.0  324.614980

xFormers
========

xFormers also improves the performance of attention modules. Although xFormers attention performs very
similarly to Flash Attention 2 due to its tiling behavior of query, key, and value, it’s widely used for LLMs and
Stable Diffusion models with the Hugging Face Diffusers library.

Installing CK xFormers 
----------------------

Use the following commands to install CK xFormers.

.. code-block:: shell
   
   # Install from source
   git clone https://github.com/ROCm/xformers.git
   cd xformers/
   git submodule update --init --recursive
   PYTORCH_ROCM_ARCH=gfx942 python setup.py install #Instinct MI300-series

PyTorch built-in acceleration
=============================

`PyTorch compilation
mode <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>`__
synthesizes the model into a graph and then lowers it to prime
operators. These operators are compiled using TorchInductor, which uses
OpenAI Triton as a building block for GPU acceleration. One advantage of
PyTorch compilation mode is that its GPU kernels are written in Python,
making modifying and extending them easier. PyTorch compilation mode
often delivers higher performance, as model operations are fused before
runtime, which allows for easy deployment of high-performance kernels.

PyTorch compilation
-------------------

To utilize the PyTorch compilation mode, specific layers of the model
must be explicitly assigned as compilation targets. In the case of LLM,
where autoregressive token decoding generates dynamically changing
key/value sizes, limiting the key/value size to a static dimension,
``max_cache_length``, is necessary to utilize the performance benefits
of the PyTorch compilation.

.. code-block:: python

   # Sample script to run LLM with the static key-value cache and PyTorch compilation
   from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
   import torch
   from typing import Optional
   import os
   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   os.environ["TOKENIZERS_PARALLELISM"] = "false"
   model_name = "NousResearch/Meta-Llama-3-8B"
   prompts = []
   
   for b in range(1):
       prompts.append("New york city is where "
   )
   
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device).eval()
   inputs = tokenizer(prompts, return_tensors="pt").to(model.device)
   
   def decode_one_tokens(model, cur_token, input_pos, cache_position):
       logits = model(cur_token, position_ids=input_pos, cache_position=cache_position, return_dict=False, use_cache=True)[0]
       new_token = torch.argmax(logits[:, -1], dim=-1)[:, None]
       return new_token
   
   batch_size, seq_length = inputs["input_ids"].shape

   # Static key-value cache
   max_cache_length = 1024
   max_new_tokens = 10
   model._setup_cache(StaticCache, batch_size, max_cache_len=max_cache_length)
   cache_position = torch.arange(seq_length, device=device)
   generated_ids = torch.zeros(batch_size, seq_length + max_new_tokens + 1, dtype=torch.int, device=device)
   generated_ids[:, cache_position] = inputs["input_ids"].to(device).to(torch.int)
   
   logits = model(**inputs, cache_position=cache_position, return_dict=False, use_cache=True)[0]
   next_token = torch.argmax(logits[:, -1], dim=-1)[:, None]

   # torch compilation
   decode_one_tokens = torch.compile(decode_one_tokens, mode="max-autotune-no-cudagraphs",fullgraph=True)
   
   generated_ids[:, seq_length] = next_token[:, 0]
   cache_position = torch.tensor([seq_length + 1], device=device)
   
   with torch.no_grad():
       for _ in range(1, max_new_tokens):
           with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
               next_token = decode_one_tokens(model, next_token.clone(), None, cache_position)
               generated_ids[:, cache_position] = next_token.int()
           cache_position += 1

.. _fine-tuning-llms-pytorch-tunableop:

PyTorch TunableOp
------------------

ROCm PyTorch (2.2.0 and later) allows users to use high-performance ROCm
GEMM kernel libraries through PyTorch's built-in TunableOp options.
This enables users to automatically pick up the best-performing GEMM
kernels from :doc:`rocBLAS <rocblas:index>` and :doc:`hipBLASLt <hipblaslt:index>` libraries during runtime.

During warm-up runs or offline profiling steps, users can create a GEMM Table
that enumerates the kernel information. During the model's run, the best-performing kernel substitutes
``torch.nn.functional.linear(input, weight, bias=None)`` with the kernel specified in the GEMM table. The
`Tunable GitHub <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/tunable/README.md>`_
page describes the options.

.. code-block:: python

   # To turn on TunableOp, simply set this environment variable
   export PYTORCH_TUNABLEOP_ENABLED=1
   
   # Python
   import torch
   import torch.nn as nn
   import torch.nn.functional as F
   A = torch.rand(100, 20, device="cuda")
   W = torch.rand(200, 20, device="cuda")
   Out = F.linear(A, W)
   print(Out.size())
   
   # tunableop_results0.csv
   Validator,PT_VERSION,2.4.0
   Validator,ROCM_VERSION,6.1.0.0-82-5fabb4c
   Validator,HIPBLASLT_VERSION,0.7.0-1549b021
   Validator,GCN_ARCH_NAME,gfx942:sramecc+:xnack-
   Validator,ROCBLAS_VERSION,4.1.0-cefa4a9b-dirty
   GemmTunableOp_float_TN,tn_200_100_20,Gemm_Rocblas_32323,0.00669595

.. image:: ../../data/how-to/llm-fine-tuning-optimization/tunableop.png
   :alt: GEMM and TunableOp
   :align: center

Learn more about optimizing kernels with TunableOp in
:ref:`Optimizing Triton kernels <fine-tuning-llms-triton-tunableop>`.

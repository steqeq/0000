.. meta::
   :description: How to fine-tune LLMs with ROCm
   :keywords: ROCm, LLM, fine-tuning, usage, tutorial, quantization, GPTQ, transformers, bitsandbytes

*****************************
Model quantization techniques
*****************************

Quantization reduces the model size compared to its native full-precision version, making it easier to fit large models
onto accelerators or GPUs with limited memory usage. This section explains how to perform LLM quantization using GPTQ,
AWQ, and bitsandbytes on AMD Instinct hardware.

.. _fine-tune-llms-gptq:

GPTQ
====

GPTQ is a post-training quantization technique where each row of the weight matrix is quantized independently to find a
version of the weights that minimizes error. These weights are quantized to ``int4`` but are restored to ``fp16`` on the
fly during inference. This can save your memory usage by a factor of four. A speedup in inference is expected because
inference of GPTQ models uses a lower bit width, which takes less time to communicate.

Before setting up the GPTQConfig in transformers, ensure the `AutoGPTQ <https://github.com/AutoGPTQ/AutoGPTQ>`_ library
is installed.

Installing AutoGPTQ
-------------------

The AutoGPTQ library implements the GPTQ algorithm.

#. Use the following command to install the latest stable release of AutoGPTQ from pip.

   .. code-block:: shell

      # This will install pre-built wheel for a specific ROCm version  
      
      pip install auto-gptq --no-build-isolation --extra-index-url https://huggingface.github.io/autogptq-index/whl/rocm573/

   Or, install AutoGPTQ from source for the appropriate ROCm version (for example, ROCm 6.1).

   .. code-block:: shell

      # Clone the source code
      git clone https://github.com/AutoGPTQ/AutoGPTQ.git
      cd AutoGPTQ
      
      # Speed up the compilation by specifying PYTORCH_ROCM_ARCH to target device 
      PYTORCH_ROCM_ARCH=gfx942 ROCM_VERSION=6.1 pip install .
      
      # Show the package after the installation 

#. Run ``pip show auto-gptq`` to print information for the installed ``auto-gptq`` package. Its output should look like
   this:

   .. code-block:: shell

      Name: auto-gptq
      Version: 0.8.0.dev0+rocm6.1
      ...

Using GPTQ with AutoGPTQ
------------------------

#. Run the following code snippet.

   .. code-block:: python

         from transformers import AutoTokenizer, TextGenerationPipeline
         from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
         base_model_name = "NousResearch/Llama-2-7b-hf"
         quantized_model_name = "llama-2-7b-hf-gptq"
         tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
         examples = [
             tokenizer(
                 "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
             )
         ]
         print(examples)

   The resulting examples should be a list of dictionaries whose keys are ``input_ids`` and ``attention_mask``.

#. Set up the quantization configuration using the following snippet.

   .. code-block:: python

      quantize_config = BaseQuantizeConfig(
          bits=4,  		# quantize model to 4-bit
          group_size=128,  	# it is recommended to set the value to 128
          desc_act=False,  
      )

#. Load the un-quantized model using the AutoGPTQ class and run the quantization.

   .. code-block:: python

      # import auto_gptq class
      from auto_gptq import AutoGPTQForCausalLM
      # load un-quantized model
      base_model = AutoGPTQForCausalLM.from_pretrained(base_model_name, quantize_config, device_map = "auto")
      base_model.quantize(examples)
      # save quantized model
      base_model.save_quantized(quantized_model_name)

Using GPTQ with Hugging Face Transformers
------------------------------------------

#. To perform a GPTQ quantization using Hugging Face Transformers, you need to create a ``GPTQConfig`` instance and set the
   number of bits to quantize to, and a dataset to calibrate the weights.

   .. code-block:: python

      from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
      
      base_model_name = " NousResearch/Llama-2-7b-hf"
      tokenizer = AutoTokenizer.from_pretrained(base_model_name)
      gptq_config = GPTQConfig(bits=4, dataset="c4", tokenizer=tokenizer)

#. Load a model to quantize using ``AutoModelForCausalLM`` and pass the
   ``gptq_config`` to its ``from_pretained`` method. Set ``device_map=”auto”`` to
   automatically offload the model to available GPU resources.

   .. code-block:: python

      quantized_model = AutoModelForCausalLM.from_pretrained(
                              base_model_name, 
                              device_map="auto", 
                              quantization_config=gptq_config)

#. Once the model is quantized, you can push the model and tokenizer to Hugging Face Hub for easy share and access.

   .. code-block:: python

      quantized_model.push_to_hub("llama-2-7b-hf-gptq")
      tokenizer.push_to_hub("llama-2-7b-hf-gptq")

   Or, you can save the model locally using the following snippet.

   .. code-block:: python

      quantized_model.save_pretrained("llama-2-7b-gptq")
      tokenizer.save_pretrained("llama-2-7b-gptq")

ExLlama-v2 support
------------------

ExLlama is a Python/C++/CUDA implementation of the Llama model that is
designed for faster inference with 4-bit GPTQ weights. The ExLlama
kernel is activated by default when users create a ``GPTQConfig`` object. To
boost inference speed even further on Instinct accelerators, use the ExLlamaV2
kernels by configuring the ``exllama_config`` parameter as the following.

.. code-block:: python

   from transformers import AutoModelForCausalLM, GPTQConfig
   pretrained_model_dir = "meta-llama/Llama-2-7b"
   gptq_config = GPTQConfig(bits=4, exllama_config={"version":2})
   quantized_model = AutoModelForCausalLM.from_pretrained(
                           base_model_name, 
                           device_map="auto", 
                           quantization_config=gptq_config)

AWQ
===

Activation-aware Weight Quantization (AWQ) doesn’t quantize all the
weights in a model. Instead, it preserves a small percentage of weights
important for LLM performance. This significantly reduces quantization
loss so you can run models with 4-bit precision without experiencing
performance degradation.

Installing AutoAWQ
------------------

AutoAWQ is a library for quantizing models using the AWQ algorithm. Use the ready-to-install wheels from
`<https://github.com/casper-hansen/AutoAWQ/releases>`__.

#. To install AutoAWQ from source, use the following in your command line.

   .. code-block:: shell

      # Clone the source code
      git clone https://github.com/casper-hansen/AutoAWQ
      cd AutoAWQ
      
      # The latest supported version of ROCm is 5.7
      python setup.py install

#. Run ``pip show autoawq`` to show information about the installed AutoAWQ package. The output should look like the
   following.

   .. code-block:: shell

      Name: autoawq
      Version: 0.2.4+rocm571
      ...

Using AWQ with AutoAWQ
-----------------------

#. To get started with AutoAWQ, use the following code.

   .. code-block:: python

      from awq import AutoAWQForCausalLM
      from transformers import AutoTokenizer
      base_model_name = "NousResearch/Llama-2-7b-hf"
      quantized_model_name = "llama-2-7b-hf-awq"
      quant_config = { 
              "zero_point": True, 
              "q_group_size": 128, 
              "w_bit": 4
      }

#. Load the un-quantized model using ``awq`` class and run the quantization:

   .. code-block:: python

      # Load model
      base_model = AutoAWQForCausalLM.from_pretrained(
          base_model_name, **{"low_cpu_mem_usage": True, "use_cache": False}
      )
      tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

      # Quantize
      base_model.quantize(tokenizer, quant_config=quant_config)

      # Save quantized model
      base_model.save_quantized(quantized_model_name)
      tokenizer.save_pretrained(quantized_model_name)

Using AWQ with Hugging Face Transformers
----------------------------------------

Transformers supports loading AWQ quantized models and performing generation.

.. code-block:: python

   from transformers import AutoModelForCausalLM, AutoTokenizer
   quantized_model_name = "llama-2-7b-hf-awq"
   model = AutoModelForCausalLM.from_pretrained(quantized_model_name, device_map="auto")

ExLlama-v2 support
------------------

Recent versions of AutoAWQ support ExLlama-v2 kernels for faster pre-fill and decoding.

.. code-block:: python

   from transformers import AutoModelForCausalLM, AutoTokenizer, AwqConfig
   quantized_model_name = "llama-2-7b-hf-awq"
   quantization_config = AwqConfig(bits=4, exllama_config={"version":2})
   quantized_model = AutoModelForCausalLM.from_pretrained(
       quantized_model_name,
       quantization_config=quantization_config,
       device_map="auto"

bitsandbytes
============

The `ROCm-aware bitsandbytes <https://github.com/ROCm/bitsandbytes>`_ library is
a lightweight Python wrapper around CUDA custom functions, in particular 8-bit optimizer, matrix multiplication, and
8-bit and 4-bit quantization functions. The library includes quantization primitives for 8-bit and 4-bit operations
through ``bitsandbytes.nn.Linear8bitLt`` and ``bitsandbytes.nn.Linear4bit`` and 8-bit optimizers through the
``bitsandbytes.optim`` module. These modules are supported on AMD Instinct accelerators.

Installing bitsandbytes
-----------------------

#. To install bitsandbytes for ROCm 6.0 (and later), use the following commands.

   .. code-block:: shell

      # Clone the github repo
      git clone --recurse https://github.com/ROCm/bitsandbytes.git
      cd bitsandbytes
      git checkout rocm_enabled

      # Install dependencies 
      pip install -r requirements-dev.txt

      # Use -DBNB_ROCM_ARCH to specify target GPU arch
      cmake -DBNB_ROCM_ARCH="gfx942" -DCOMPUTE_BACKEND=hip -S .

      # Install 
      python setup.py install

#. Run ``pip show bitsandbytes`` to show the information about the installed bitsandbytes package. Its output should
   look like the following.

   .. code-block:: shell

      Name: bitsandbytes
      Version: 0.44.0.dev0
      ...

Using bitsandbytes primitives
-----------------------------

To get started with bitsandbytes primitives, use the following code a reference.

.. code-block:: python

   import bitsandbytes as bnb
   
   # Use Int8 Matrix Multiplication
   bnb.matmul(..., threshold=6.0)
   
   # Use bitsandbytes 8-bit Optimizers
   adam = bnb.optim.Adam8bit(model.parameters(), lr=0.001, betas=(0.9, 0.995))

Using bitsandbytes with Hugging Face Transformers
-------------------------------------------------

To load a Transformers model in 4-bit, set ``load_int_4bt=true`` in ``BitsAndBytesConfig``.

.. code-block:: python

   from transformers import AutoModelForCausalLM
   from bitsandbytes import BitsAndBytesConfig
   
   base_model_name = "NousResearch/Llama-2-7b-hf"
   quantization_config = BitsAndBytesConfig(load_in_4bit=True)
   bnb_model_4bit = AutoModelForCausalLM.from_pretrained(
           base_model_name, 
           device_map="auto", 
           quantization_config=quantization_config)
   
   # check the memory footprint with get_memory_footprint method
   print(bnb_model_4bit.get_memory_footprint())

To load a model in 8-bit for inference, use the ``load_in_8bit`` option.

.. code-block:: python

   from transformers import AutoModelForCausalLM, AutoTokenizer
   from bitsandbytes import BitsAndBytesConfig
   
   base_model_name = "NousResearch/Llama-2-7b-hf"
   
   tokenizer = AutoTokenizer.from_pretrained(base_model_name)
   quantization_config = BitsAndBytesConfig(load_in_8bit=True)
   tokenizer = AutoTokenizer.from_pretrained(base_model_name)
   bnb_model_8bit = AutoModelForCausalLM.from_pretrained(
           base_model_name, 
           device_map="auto", 
           quantization_config=quantization_config)
   
   prompt = "What is a large language model?"
   inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
   generated_ids = model.generate(**inputs)
   outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


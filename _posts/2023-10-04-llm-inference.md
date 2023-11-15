---
layout: post
title: Different techniques for optimizing LLM inference
date: 2023-10-04 00:09:00
description: 
tags: dev
categories: NLP, LLM Inference
---

---


### Model Optimization

+ <b>[ONNX model format:](https://huggingface.co/docs/optimum/onnxruntime/concept_guides/onnx)</b> a common file format used to store deep learning models, including LLMs,that is compatible with various frameworks such as PyTorch, TensorFlow. It represents the model as a computational graph that can be optimized for efficient inference by dedicated Runtime through graph operators like redundant node eliminations (e.g. dropout is not utilized during inference, hence eliminated), node fusion (e.g. Conv layer and batchNorm layer can be merged into one computing node), constant folding (i.g. recognize and contant expressions at compile time insteading of computing them at runtime), etc. In addition, ONNX Runtime supports also model static and dynamic quantization.

    ONNX Runtime provides 2 accelerator to speed up the inference (with several limitations) on NVIDIA GPU devices:

    + CUDA Execution Provider: not possible to run quantized model.
        
    + TensorRT Execution Provider: only support static quantization, offer its own graph optimization techniques rather than inheriting from ONNX Runtime.

+ <b>Quantization</b>:

    + [bitsandbytes](https://github.com/TimDettmers/bitsandbytes): support 8-bit and 4-bit zero-point and absmax quantization. As 8-bit and 4-bit represent a very limited range, quantizing 32-bit or 16-bit big values (i.e. outliers) can cause a degradation in the inference. bitsandbytes addresses this issue with mixed-precision decomposition. The idea is: in matrix multiplication of (hidden feature, weights), a feature column $$i$$ that contain outliers is multiplied with the weight row $$i$$ in the original precision (FP32, FP16 or BF16), otherwise, the features and weights are quantized, multiplied and dequantized. 

    + [GPTQ:](https://huggingface.co/blog/gptq-integration) different from bitsandbytes, in GPTQ, only weights are quantized into 8,4 or 3 bits, activations are retained in float16 and during the inference, the actual computation is performed in float16 rather than in smaller bits. The key idea of GPTQ is to find a quantized version $$\hat{W}$$ of model weights $$W$$ by minimizing the MSE loss $$\| WX - \hat{W}X \|$$ on a reference calibration dataset.

### OS Optimization

+ <b>[Batching:](https://www.anyscale.com/blog/continuous-batching-llm-inference)</b> in order to better utilize the hardware (i.e. GPU memory) during the inference processing multiple requests or samples at once in a batch (i.e. batching) is a good pratice.
    
    + Static batching: it has to wait for incoming requests that fulfill a predefined batch size before processing. Also, GPU resources reserved for a batch is blocked until the last request in the batch is finished, leading to inefficient GPU utilization when requests have have significantly different execution times.

    + Continous batching: to tackle the issue of static batching, continous batching allows to drop finished requests from the batch, and replace them with new requests without need to wait for the completion of every other requests in the batch.

+ <b>[Paged Attention:](https://blog.vllm.ai/2023/06/20/vllm.html)</b> manages more effectively attention keys and values (KV cache). Instead of keeping all attention keys/values of a sequence in a contiguous GPU memory block, PagedAttention splits the KV cache into blocks, each block containing the keys and values for a number of tokens can be stored flexible in non-contiguous memory blocks. When computing the attention for the whole sequence, involved KV blocks are retrived via a block mapping table. Moreover, by using an universal block table, multiple requests that share the same input prompt can share prompt's KV(s) rather than computing them individually for each request, resulting in efficient memory utilization.

+ <b>[Flash Attention:](https://blog.vllm.ai/2023/06/20/vllm.html)</b> accelerates the attention computation and reduces its memory footprint without zero accuracy loss, especially for long context, with two techniques: (i) avoid computing huge $$QK^T$$ matrix at once by breaking down $$K$$ into chunks $$K_i$$, perform each $$QK_i^T$$, then combine results; (ii) $$Q$$ and $$K_i$$ are moved from GPU's memory to SRAM in order to compute attention in SRAM, thereby reducing memory reads/writes.

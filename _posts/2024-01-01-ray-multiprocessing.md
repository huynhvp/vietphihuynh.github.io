---
layout: post
title: I find Ray a powerful framework for parallel computing
date: 2024-01-01 00:09:00
description: 
tags: dev
categories: multiprocessing, ray
---

One strategy to speed up or scale a machine learning workflow is parallel/distributed processing. In python, the [multiprocessing](https://docs.python.org/3/library/multiprocessing.html) module can serve as a solution for this purpose. However, it falls short (and can even harm overal performance) for parallel functions that requires heavy workloads or costly initialization due to data copying, moving and overhead input serialization/deserialization.

Meanwhile, [Ray](https://docs.ray.io/en/latest/index.html) is perfectly suited to such scenarios. Let's work on two toy examples to illustrate that.

>
Two core concepts (among others) of Ray that make it powerful in distrubed programming are: 
+ [Task](https://docs.ray.io/en/latest/ray-core/key-concepts.html#tasks): like an asynchronous function that can be executed in a seperate process or a remote machine.
+ [Actor](https://docs.ray.io/en/latest/ray-core/key-concepts.html#actors): like an asynchronous stateful class that can run in a seperate process or remotely together with its own methods. Particularly, other actors and tasks from different processes can acess and mutate actor's states.

---
<b>Table of Contents</b>
* TOC []
{:toc}





#### 1. Parallelize a bundle of matrix multiplication functions

Let's parallelize a set of functions described by $$f_i = x*y_i$$ where $$x$$ is a fixed 10240x10240 float matrix (~800MB), representing heavy input, and $$y_i$$ denotes a variable 10240x1024 float matrix for each function $$f_i$$.


<b>With Multiprocessing:</b>

```python
import time
import tracemalloc
from multiprocessing import Pool

import numpy as np
import psutil

num_cpus = psutil.cpu_count(logical=False)  # 8
num_workers = num_cpus // 2
pool = Pool(num_workers) # multiprocessing pool


np.random.seed(1234)
x = np.random.randn(10240, 10240)  # x takes 800MB

def task(x, y):  # noqa: D103
    z = np.matmul(x, y)
    return z

def run_multiple_tasks_in_parallel(i_trial):
    np.random.seed(i_trial)
    y_s = [np.random.rand(10240, 128) for _ in range(num_workers)]  # each y takes 10MB
    results = pool.starmap(task, zip([x] * len(y_s), y_s, strict=True))
    return results

if __name__ == "__main__":
    # benchmark
    num_trials = 10
    start_time = time.perf_counter()

    # tracemalloc.start()
    for i_trial in range(num_trials):
        run_multiple_tasks_in_parallel(i_trial)
    # current_mem, peak_mem = tracemalloc.get_traced_memory()
    # tracemalloc.stop()

    end_time = time.perf_counter()
    print(
        f"Avg Time: {(end_time - start_time)/num_trials:.2f} (s)"
    )  # while benchmarking time, disable mem_usage to avoid additional calculation.
    # print(f"Peak memory: {peak_mem/(1024*1024):.2f} (MB)")
```

```bash
Avg Time: 7.31 (s)
Peak memory: 1770.71 (MB)
```

<b>With Ray:</b>

```python
import ray

num_cpus = psutil.cpu_count(logical=False)  # 8
num_workers = num_cpus // 2  # max parallel tasks
ray.init(num_cpus=num_workers)  # init ray

np.random.seed(1234)
x = np.random.randn(10240, 10240)  # x takes 800MB
x_ref = ray.put(x)  # put x in ray's object store and return its reference.

@ray.remote  # convert func to ray's remote task
def task(x, y):  # noqa: D103
    z = np.matmul(x, y)
    return z

def run_multiple_tasks_in_parallel(i_run):
    np.random.seed(i_run)
    y_s = [np.random.rand(10240, 128) for _ in range(num_workers)]  # each y takes 10MB
    ray_task_list = [task.remote(x_ref, y) for y in y_s]  # pass reference of x instead of x itself
    results = ray.get(ray_task_list)
    return results

```
```bash
Avg Time: 0.95 (s)
Peak memory: 44.38 (MB)
```

**<span style="color:green"><b>Clearly,</b></span>** Ray is much faster and much more memory-efficient than Multiprocessing. This is due to the fact that in Multiprocessing, each process worker has to copy and pass expensive input data (i.e. $$x$$) from main process, which ends up with overhead serialization/deserialization and high memory usage. On the contrary, in Ray, the main process puts the fixed matrix $$x$$ in a shared object store and pass the reference of $$x$$ to each worker. This reduces memory usage as each worker now uses the same object $$x$$, without duplication. Morever, for object of primitive datatypes, such as numpy array, Ray avoids serializing them, allowing process workers to read them directly without deserialization, leading to significant performance gains.

#### 2. Parallelize a bunch of Named Entity Recognition (NER) models

Considering the scenarios where a server is asked to tag named entities in a batch of text. The server calls on workers, each of which load a NER model and process a text in batch. When a batch is completed, another batch arrives and the workers continue their works.

<b>With Multiprocessing:</b>

```python
import time
import tracemalloc
from multiprocessing import Pool

import numpy as np
import psutil
import spacy

num_cpus = psutil.cpu_count(logical=False)  # 8
num_workers = num_cpus // 2
pool = Pool(num_workers) # multiprocessing pool

def task(text):
    ner = spacy.load("en_core_web_sm") # load NER model
    entities = ner(text)
    return entities

def run_multiple_tasks_in_parallel(i_batch):
    np.random.seed(i_batch)
    text_s = ["Paris is the capital of France."] * num_workers
    results = pool.map(task, text_s)
    return results

if __name__ == "__main__":
    # benchmark
    num_batches = 10
    start_time = time.perf_counter()

    # tracemalloc.start()
    for i_batch in range(num_batches): # one batch is completed, another arrives.
        run_multiple_tasks_in_parallel(i_batch)
    # current_mem, peak_mem = tracemalloc.get_traced_memory()
    # tracemalloc.stop()

    end_time = time.perf_counter()
    print(
        f"Total Time for processing {num_batches} batches of texts: {end_time - start_time:.2f} (s)"
    )  # while benchmarking time, disable mem_usage to avoid additional calculation.
    # print(f"Peak memory: {peak_mem/(1024*1024):.2f} (MB)")
```

```bash
Total Time for processing 10 batches of texts: 38.17 (s)
Peak memory: 143.63 (MB)
```

<b>With Ray:</b>

```python
import ray

num_cpus = psutil.cpu_count(logical=False)  # 8
num_workers = num_cpus // 2
ray.init(num_cpus=num_workers)

@ray.remote
class NER:
    def __init__(self):
        self.ner = spacy.load("en_core_web_sm")  # load NER model

    def tag(self, text):  # named entity tag function
        entities = self.ner(text)
        return entities

# creat ray workers via actor, the ner models are loaded once at actor's construction time
ner_actors = [NER.remote() for _ in range(num_workers)]

def run_multiple_tasks_in_parallel(i_batch):
    np.random.seed(i_batch)
    y_s = ["Paris is the capital of France."] * num_workers
    results = ray.get([actor.tag.remote(y) for actor, y in zip(ner_actors, y_s)])
    return results
```

```bash
Total Time for processing 10 batches of texts: 8.73 (s)
Peak memory: 144.62 (MB)
```

**<span style="color:green"><b>Worker processes</b></span>** in Multiprocessing.Pool are stateless, thus, for every `pool.map` call for every batch, the NER models need to be reloaded. Meanwhile, Ray's actors are stateful, NER models are loaded only once at actor's construction time (i.e. `__init__`function). Future batches are then processed by just calling `tag` function. This explains the outperformance of Ray over Multiprocessing. Additionally, in term of memory usage, both frameworks observe similary memory peaks, as the task does not involve any large data objects.
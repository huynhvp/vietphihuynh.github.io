---
layout: post
title: Cheat Sheet of NLP Practitioner
date: 2022-11-13 10:09:00
description: 
tags: research
categories: NLP
toc-title: "INDICE"
---

---

I love reading research papers, blogs, tutorials, etc, that aligns with my domains of interest. By reading and practicing, I learn not only interesting ideas, new methods that keep me up to date with the most recent trends/advances but also best practices that just make me better and better. I find it useful to write all them down (briefly) in an unified place, potentially aiming at a systematic review and insights gaining. For this goal, I am actively maintaining this blog post.

---

<b>Table of Contents</b>
* TOC [aerer]
{:toc}

### <b>1. Best Practices</b>
#### <b>1.1. Training/Fine-Tuning recipes</b>

<b>2020</b>

- [Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks](https://aclanthology.org/2020.acl-main.740) (Gururangan et al., ACL 2020): 

    Before fine-tuning, continue pre-training a general pretrained language model (PLM) on in-domain unlabeled data (*domain-adaptive pretraining*) or task-specific unlabeled data (*task-adaptive pretraining*) can improve the performance of downstream tasks.

<b>2019</b>

- [When does label smoothing help?.](https://arxiv.org/abs/1906.02629) (Müller et al., NeurIPS 2019).

    Optimizing cross entropy loss with hard targets (i.e. one-hot encoding labels) can make the model predict a training sample too confidently where the logit predicted for true label is very large comparing with ones predicted for other labels, as a consequence, the softmax function will generate probabilities with huge gap (e.g. 0.99 for target label and ~0.0 for other labels). To alleviate this issue, one solution is to increase the *temperature T* to smooth out soft-max probabilities. Another solution is: instead of training with one-hot encoded label (e.g. [1, 0, 0]), we use soft label (e.g. [0.9, 0.05, 0.05]) by re-weighing labels with a small added value playing as noise. <b>Note:</b> we shoud not distill knowledge from a teacher model which is trained with label smoothing since it cause accuracy degradation. 

<br>

### <b>2. Topics</b>
#### <b>2.1. Neural Text Generation </b>
##### <b>2.1.1 Decoding methods </b>

<b>2022</b>

- [A Contrastive Framework for Neural Text Generation](https://arxiv.org/pdf/2202.06417.pdf) (Su et al., NeurIPS 2022).

    Aiming at avoiding repetition patterns while maintaining semantic coherence in generated text, <b>constrastive search</b> introduces a *degeneration penalty* into the decoding objective. This *degeneration penalty* compares the cosine similarity between a token at current decoding step and all generated tokens at previous decoding steps. The closer the token is to precedent decoded text (more likely leading to repetition), the larger the penalty it receives.

#### <b>2.2. Sentence Embedding </b>

<b>2021</b>

- [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://aclanthology.org/2021.emnlp-main.552) (Gao et al., EMNLP 2021).

    Contrastive learning is employed to learn the sentence embedding with a single encoder in unsupervised manner. They use dropout for the generation of positive samples. Specifically, an input sentence is fed to the LM *twice* with two different dropout masks that will generate a positive pair of sentence representations for the training. Two take-away messages: (i) dropout as data augmentation for text, (ii) contrastive learning helps to evenly distribute learned representations in the embedding space (*isotropy*).

#### <b>2.3. Entity Linking </b>

<b>2021</b>

- [GENRE: Autoregressive Entity Retrieval](https://arxiv.org/pdf/2010.00904.pdf) (De Cao et al., ICLR 2021).

    Very interesting entity linker that casts the entity linking problem as a text-to-text problem and employs a seq2seq model (i.e. BART) to address it.

    Example:
    ```console
    Encoder: In 1503, Leonardo began painting the Mona Lisa
    Decoder: In 1503, [Leonardo](Leonardo da Vinci) began painting the [Mona Lisa](Mona Lisa)

    where [X](Y) : X is the mention, and Y is the entity label (aka. entity identifier) that represents X.
    ```

    Importantly, they perform the inference with constrained beam search to force the decoder to generate the valid entity identifier. Specifically, at a decoding step $$t$$, the generation of the next token $$x_t$$ is conditioned on previous ones $$x_1,..., x_{t-1}$$ such that $$x_1,..., x_{t-1}, x_{t}$$ is a valid n-gram of an entity identifier.

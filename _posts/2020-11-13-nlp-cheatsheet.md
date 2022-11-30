---
layout: post
title: Cheat Sheet of NLP Practitioner
date: 2022-11-13 10:09:00
description: 
tags: research
categories: NLP, AI
---

---

I love reading research papers, blogs, tutorials, etc, that aligns with my domains of interest. By reading and practicing, I learn not only interesting ideas, new methods that keep me up to date with the most recent trends/advances but also best practices that just make me better and better. I find it useful to write all them down (briefly) in an unified place, potentially aiming at a systematic review and insights gaining. For this goal, I am actively maintaining this blog post.

---

<b>Table of Contents</b>
* TOC []
{:toc}

### <b>1. Best Practices</b>
#### <b>1.1. Training/Fine-Tuning recipes</b>

<b>2021</b>

- [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://aclanthology.org/2021.acl-long.353.pdf) (Li et al., ACL 2021)

    Traditional fine-tuning of a LM model for a downstream task involves modifying all the model parameters, consequently, a single set of parameters  can just work best for a single task. Inspired by prompting, <b>prefix-tuning</b> freezes the LM parameters and instead prepend to it a sequence of task-specific vectors $$P_{\theta}$$ (aka. *prefix*): $$[P_{\theta}; LM_{\phi}]$$ that represent the downstream task, we optimize solely the *prefix* $$P_{\theta}$$ using the task's data to steer the LM to the task.

    Prefix-tuning brings some advantages:

    - A single LM is reused across different downstream tasks since its parameters are kept intact $$\rightarrow$$ efficient storage.
    - Only the prefix vector corresponding to the downstream task need to be optimized $$\rightarrow$$ lightweight fine-tuning: much fewer parameters w.r.t. LM. 
    - Prefix-tuning can outperform full fine-tuning in low-data setting and have better generalization.

<b>2020</b>

- [Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks](https://aclanthology.org/2020.acl-main.740) (Gururangan et al., ACL 2020): 

    Before fine-tuning, continue pre-training a general pretrained language model (PLM) on in-domain unlabeled data (*domain-adaptive pretraining*) or task-specific unlabeled data (*task-adaptive pretraining*) can improve the performance of downstream tasks.

<b>2019</b>

- [When does label smoothing help?.](https://arxiv.org/abs/1906.02629) (Müller et al., NeurIPS 2019).

    Optimizing cross entropy loss with hard targets (i.e. one-hot encoding labels) can make the model predict a training sample too confidently where the logit predicted for true label is very large comparing with ones predicted for other labels, as a consequence, the softmax function will generate probabilities with huge gap (e.g. 0.99 for target label and ~0.0 for other labels). To alleviate this issue, one solution is to increase the *temperature T* to smooth out soft-max probabilities. Another solution is: instead of training with one-hot encoded label (e.g. [1, 0, 0]), we use soft label (e.g. [0.9, 0.05, 0.05]) by re-weighing labels with a small added value playing as noise. <b>Note:</b> we shoud not distill knowledge from a teacher model which is trained with label smoothing since it cause accuracy degradation. 

#### <b>1.2. Data Augmentation</b>

<b>2022</b> 

- <b>From zero-shot to few-shot Text Classification with [SetFit](https://arxiv.org/pdf/2209.11055.pdf)</b>
    
    SetFit is a few-shot text classifier (e.g. sentiment analysis) based on [Sentence Transformer](https://arxiv.org/abs/1908.10084). Speaking of its performance,
    >  With only 8 labeled examples per class on the Customer Reviews (CR) sentiment dataset, SetFit$$_{MPNET}$$ (110M parameters) is competitive with fine-tuning RoBERTa Large (355M parameters) on the full training set of 3k examples 🤯. (Source: https://huggingface.co/blog/setfit)
    
    In zero-shot setting, we can generate some very simple samples for each classification label (e.g. 8 samples per label) to make it a few-shot learning problem. For example, in the sentiment analysis task, using template "This sentence is about {}", a positive sample for label "joy" can be "This sentence is about joy", for label "sadness" can be "This sentence is about sadness", etc.

<b>2021<b>

- <b>Dropout</b> [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://aclanthology.org/2021.emnlp-main.552) (Gao et al., EMNLP 2021)

    An input sentence is fed to the LM *twice* with two different dropout masks that will generate a positive pair of sentence representations for the training.

<b>2016</b>

- <b> Back Translation</b> [Improving Neural Machine Translation Models with Monolingual Data](https://aclanthology.org/P16-1009) (Sennrich et al., ACL 2016) 

    Given a text in a known language, we translate it into some other languages and then translate it back to the original language. This will generate synthetic texts that syntactically differ from the input text but have similar semantics. For example, the English sentence "I love watching move" is translated into French: "J'aime regarder un film" then mapped back to English: "I like to watch a movie".

#### <b>1.3. Text scoring function</b>

The likelihood of a text $$y=y_1, y_2,...,y_n$$ (where $$y_i$$ is a token in the vocabulary) of length $$n$$ given an input text $$x$$,  is given by a LM:

$$p(y \mid x) = \prod_{i=1}^{n} p(y_i \mid x, y_{i-1}...y_1)$$

However, in the context of scoring function, the likelihood $$p(y \mid x)$$ is not widely used to compare the text $$y$$ with other texts $$y'$$ given $$x$$. Instead, the *length-normalized* log-likelihood has been standard for this end. 

$$score \; (y \mid x) = \frac{log \; p(y \mid x)}{n} = \frac{\sum_{i=1}^{n} log \; p(y_i \mid x, y_{i-1}...y_1) }{n}$$

<b>2021</b>

- [Surface Form Competition: Why the Highest Probability Answer Isn’t Always Right](https://arxiv.org/pdf/2104.08315.pdf) (Holtzman et al., EMNLP 2021)

    This paper investigates an very interesting problem of text scoring function used to determine a prediction $$y$$ for an input $$x$$ with LM: <b> surface form competition </b>. Specifically, given $$x$$, there could be many relevant $$y$$(s) that differ from their surface forms but share the same underlying concept in the context of $$x$$. For example, if $$x$$ is "Which is the richest country in the world", then $$y$$ could be "USA", "United States", "U.S.A" or even "U.S of A". All those answers should receive high score, however, since they come from the same finite probability mass function $$p(y \mid x)$$, they compete each other for how much probability they could get. Due to the different level of popularity of each answer $$y$$ in the training corpus, the model tends to allocate much more probability mass to popular "United States" or "USA", which consequently decrease the amount for rare "U.S of A".
    
    <b>Solution</b> Rather than calculating the ranking score $$score \; (y \mid x)$$  via $$p(y \mid x)$$ which make solutions $$y$$ compete each other, the <b>Pointwise Mutual Information (PMI)</b> is leveraged to evaluate the relevance between the input $$x$$ and the output $$y$$:

    $$score \; (y \mid x) = \text{PMI}(x, y) = log \frac{p(x,y)}{p(x) \times p(y)} = log \frac{p (x \mid y)}{p(x)}$$

    While $$p (x)$$ is constant w.r.t $$y$$ and the probability of surface form $$p (y)$$ is factored out in $$\text{PMI}(x, y)$$, the ranking of a solution $$y$$ relies solely on $$p (x \mid y)$$ that does not cause the competition between different $$y$$.

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

#### <b>2.3. Entity Linking & Disambiguation</b>

<b>2021</b>

- [GENRE: Autoregressive Entity Retrieval](https://arxiv.org/pdf/2010.00904.pdf) (De Cao et al., ICLR 2021).

    Very interesting entity retriever that casts the entity linking problem as a text-to-text problem and employs a seq2seq model (i.e. BART) to address it.

    Example:
    ```console
    Encoder: In 1503, Leonardo began painting the Mona Lisa
    Decoder: In 1503, [Leonardo](Leonardo da Vinci) began painting the [Mona Lisa](Mona Lisa)

    where [X](Y) : X is the mention, and Y is the entity label (aka. entity identifier) that represents X.
    ```

    Importantly, they perform the inference with constrained beam search to force the decoder to generate the valid entity identifier. Specifically, at a decoding step $$t$$, the generation of the next token $$x_t$$ is conditioned on previous ones $$x_1,..., x_{t-1}$$ such that $$x_1,..., x_{t-1}, x_{t}$$ is a valid n-gram of an entity identifier.

#### <b>2.4. Probing Knowledge from Language Model</b>
##### <b>2.4.1 Knowledge Retriever + Language Model </b>

<b>Overview:</b>

Knowledge retriever aims at retrieving support passage (documents) that can help to explain the knowledge probed from LM.

<b>2021</b>

- [Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering](https://arxiv.org/abs/2007.01282) (Izacard et al., EACL 2021)

    <b>Retrieved evidence fusion in decoder (*Fusion-in-Decoder*).</b>

    To address the Open Domain Question Answering, firstly, an independent knowledge retriever is leveraged to retrieve supporting passages for the input question, then, a seq2seq model (T5) takes as input the combination of the input question and supporting passages to produce the answer. Specifically, each retrieved passage concatenated with the input question is independently encoded by the encoder and their representations are merged together before sending to the decoder, in this way, the decoder can attend over the whole set of retrieved potential evidences and rely on them to generate the answer. There are two advantages of this *fusion-in-decoder* method:

     - Avoid encoding all retrieved passages and the input question in one place which is very costly due to significant lengths of the passages. 
     - Thanks to the first point, we can efficiently increase the number of retrieved support passages,
     leading to the higher accuracy in the answer.

<b>2020</b>

- [REALM: Retrieval-Augmented Language Model Pre-Training](https://arxiv.org/pdf/2002.08909.pdf) (Guu et al., ICML 2020)

    <b>Knowledge Retriever jointly pre-trained with LM.</b>

    BERT-style LM is pre-trained to denoise a corrupted input sentence $$\hat{x}$$ by predicting the masked tokens [MASK] in $$\hat{x}: p(x| \hat{x})$$. For example, given "The [MASK] is the currency of the United
Kingdom" as $$\hat{x}$$, then the answer for [MASK] is "pound". REALM makes the prediction more interpretable by first retrieving possibly helpful documents $$z$$ ($$z$$ plays as latent variable) from a knowledge source $$Z$$ and using them (as evidences) to support the prediction of [MASK], as following:

    $$p(x \mid \hat{x}) = \sum_{z \in Z}{ p (x | \hat{x}, z ) * p (z | \hat{x}) }$$

    $$ p (x \mid \hat{x}, z ) $$ helps to inform which documents $$z$$ contribute the most to [MASK] tokens. The <b>knowledge retriever</b> $$ p_{\theta}(z \mid \hat{x}) $$ and <b>knowledge-augmented encoder</b> $$ p_{\phi}(x \mid \hat{x}, z )$$ are modelled separately using two different BERT$$_{\theta}$$ and BERT$$_{\phi}$$. Document $$z$$ is represented by its title and body. $$ p_{\theta}(z \mid \hat{x}) $$ involves the cosine similarity between the sentence embedding produced by BERT$$_{\theta}$$ of $$\hat{x}$$ and $$z$$. During the pre-training, the marginal $$p(x \mid \hat{x})$$ requires a summation over all documents $$z$$ in $$Z$$ which is very costly. Also, as $${\theta}$$ changes every training step, hence the embeddings <b>Emb(z)</b> of all documents $$z$$ in $$Z$$ need to be recalculated every step $$\rightarrow$$ sound impossible. To deal with these issues, REALM proposes two training strategies
     - $$ p_{\theta}(z \mid \hat{x}) $$ is marginalized over only top-K documents $$z$$ instead of all. Top-K relevant documents $$z$$ w.r.t. input $$\hat{x}$$ can be efficiently performed by Maximum Inner Product Search (MIPS) algorithm where the embeddings of $$z$$(s) are pre-computed and pre-indexed.
     - The <b>Emb(z)</b> are freezed for an amount of time and are only re-calculated every several hundred update step.

##### <b>2.4.2 Automated Knowledge Base Construction with Language Model</b>

[An overview](https://www.mpi-inf.mpg.de/fileadmin/inf/d5/teaching/ss22_akbc/8_LMs_and_KBs.pdf)

<b>Remarkable Challenges:</b>
- LM is not trained to assert factual knowledge, but to predict masked tokens/next tokens. So when it is seen predicting a true fact, is it because of the knowledge it learned or just the correlation with what it learned ([educated guess](https://aclanthology.org/2021.acl-long.146/))

- LM Probability is not a proof of veracity but rather relates to the likelihood of a token over others during the pre-training $$\rightarrow$$ LM should know its limit when answering something (e.g. chose to answer "Unknown" or "No" instead of attempting to say anything)

<b>2022</b>

- [Task-specific Pre-training and Prompt Decomposition for Knowledge Graph Population with Language Models](https://lm-kbc.github.io/static/papers/paper_2.pdf) (Li et al., LM-KBC@ISWC 2022 Challenge)

    This work continues to pre-train BERT with task-specific data to make it familiar with the task. How ? triples *<sub, rel, obj>* are verbalized into a sentence using a prompt template of *rel*. As the task is object prediction, the object or surround words in the sentence are masked and the LM is asked to predict them. Large dataset is necessary for pre-training, hence, they leverage Wikidata for data augmentation where they generate KG triples that have same relations as provided training relations). However, they discover later that the accuracy does not clearly relate to data size but the property of relation (see below).
    - Prompt generation: they curate a set of prompts for a relation both in manual and automatic way. In manual way, they explicitly append the type of the subject into the prompt, such as "The musician [SUBJ] plays [OBJ]" for relation "PersonInstrument". In automatic way, they employ two methods from [How Can We Know What Language Models Know?](https://arxiv.org/pdf/1911.12543.pdf). However, in contrast to [How Can We Know What Language Models Know?](https://arxiv.org/pdf/1911.12543.pdf), this paper shows that an ensemble of automatically-generated prompts is not better than a single manual-curated one.
    - Prompt decomposition: a relation can have diverse domain and diverse range. For example, considering the relation "StateSharesBorderState", its domain can include "Andalusia"-is a autonomous community or "Hebei" - a province. To better distinguish the type of the subject and probe more relevant knowledge from LM, two prompts are performed:
      - ask for subject type: e.g. e "[SUBJ], as a place, is a [TYPE]".
      - inject the subject type into the prompt of the relation: e.g. "[SUBJ] [TYPE] shares border with [MASK] [TYPE]". 


<b>2020</b>

- [How Can We Know What Language Models Know?](https://arxiv.org/pdf/1911.12543.pdf) (Jiang et al., TACL 2020)

    Knowledge in LM can be probed by asking the LM fill in the blanks of prompts such as "CR7 plays for ___". This prompt-based method can only measure the lower bound of amount of knowledge contained in LM as there is no single prompt that works best for all instances of a relation (depending on what LM sees during its pre-training). To predict a missing object in a KB triple $$tpl$$: *<sub, rel, ?>*, $$tpl$$ is converted into a cloze-style prompt $$t_r$$ that semantically expresses the relation *rel* and let the LM predict the object by filling the blank in $$t_r$$. No prompt fits all, they propose two ways to generate a set of prompts for each relation $$r$$:
     - *Mining-based generation*: <b>(i)</b> collecting sentences that contain both subject and object of a given relation $$r$$, words between subject and object can be viewed as a representation of $$r$$; <b>(ii)</b> if there is no meaningful middle words, sentence is analyzed syntactically, a prompt for $$r$$ can be generated from the dependency tree.
     - *Paraphrasing-based generation*: starting from an initial prompt $$p$$ for $$r$$, $$p$$ is paraphrased into other $$p'$$ semantically similar. For example, if $$r$$ == "*hasName*" has a prompt $$p$$ == "*x is named as y*" then $$p'$$ could be "*y is a name of x*". Back-translation is a prevailing method for paraphrasing.

    <br>
    <b>Thoughts</b>: 
     - Blank in cloze-style prompt: how does LM know if ___ is single-token and multi-tokens (this work defaults single token).
     - Domain and Range of a relation are ignored: a relation can appear under many different situations. A prompt is suitable for a situation but could turn out to be strange for other situations.

#### <b>2.5. Domain-specific Language Model </b>

<b>2021</b>

- [Adapt-and-Distill: Developing Small, Fast and Effective Pretrained Language Models for Domains](https://aclanthology.org/2021.findings-acl.40.pdf) (Yao et al., ACL Findings 2021)

    To adapt a general domain LM to a specific domain, it is necessary to augment the original vocabulary with domain-specific subwords or terms (original vocabulary is kept intact). The paper proposes a simple method to determine domain-specific tokens to add to the vocabulary.

    It assumes that each subword $$x_i$$ is independent of another and it is assigned a probability $$p(x_i)$$ equal to its frequency in the corpus:

    $$\forall i \; x_i \in \mathcal{V}, \; \sum_{x_i \in \mathcal{V}} p(x_i)  = 1$$ where $$\mathcal{V}$$ is the vocabulary.

    and the log probability of a sentence $$x$$ consisting of a subword sequence $$x = (x_1,...,x_M)$$ is given by: $$ P(x) = log \prod_{i=1}^{M} p(x_i) = \sum_{i=1}^{M} log \; p(x_i)$$

    Given a domain-specific corpus D consisting of $$\mid$$D$$\mid$$ sentences, the likelihood of D is calculated as: $$ P(D) = \sum_{x \in D} log \; P(x)$$.

    The original vocabulary is iteratively enriched with subwords taken from domain corpus D. At the time step $$i$$, a subset of subwords with highest frequency in D is added to the vocabulary,  which helps to improve the likelihood $$P(D)$$. The procedure continues if the likelihood gain w.r.t. previous time step $$i-1$$ is higher than a threshold $$\delta$$: $$\frac{P_{i} (D) - P_{i-1} (D)}{P_{i-1} (D)} > \delta$$

- [UDALM: Unsupervised Domain Adaptation through Language Modeling](https://aclanthology.org/2021.naacl-main.203.pdf) (Karouzos et al., NAACL 2021)
    
    This method adapts a general pretrained LM to the target domain distribution in a simple strategy consisting of three steps:

    - Pre-training LM on general corpus using MLM objective.
    - Continue the pre-trainining on target domain corpus using MLM objective
    - Perform simultaneously/interleavely two supervised fine-tuning task: (i) a supervised task on labelled source domain data (e.g. classification) (ii) MLM task on target domain data. The idea is to avoid the <b>catastrophic forgetting</b> while adapting the general LM to target domain:
    
        $$Loss = \lambda Loss_{classification \; task} + (1-\lambda) Loss_{MLM \; task}$$.

        During this process, the samples from two tasks are interleaved in a batch and are fed to the BERT encoder. The value of $$\lambda$$ is determined by the proportion of samples of the classification task (i) in the batch. 

<b>2020</b>

- [BioMegatron: Larger Biomedical Domain Language Model](https://aclanthology.org/2020.emnlp-main.379.pdf) (Shin et al., EMNLP 2020)

    BioMegatron is a Megatron-LM pretrained on PubMed dataset and/or others general corpus for Biomedical domain. 

    The paper studies the impact of several factors on the performance of both general LM and domain-adapted LM on 3 applications: NER, RE and Q/A in Biomedical domain.

    - Domain-specific vocabulary is important for NER and RE task as general-vocabulary breaks domain named-entities into sub-words.
    - Q/A: (i) BioMegatron with <b>Bio-vocab</b> finetuned on general SQUAD then on BioASQ results poor results on BioASQ. (ii) larger models tend to perform better.
    - Domain Transfer & Generalization: (i) NER: general LLM with general vocabulary if pre-trained sufficiently on domain-specific corpus can be as good as a LM pre-trained only domain corpus only with general vocabulary. (ii) Q/A: large general LM fine-tuned on BioASQ does not mean better performance. (iii) General-domain Q/A: large BioMegatron performs better than small general LM on general-domain Q/A.
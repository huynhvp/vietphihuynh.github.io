---
layout: post
title: Cheat Sheet of NLP Practitioner
date: 2022-12-19 10:09:00
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

<b>2022</b>

- [Improving Language Models by Retrieving from Trillions of Token](https://proceedings.mlr.press/v162/borgeaud22a/borgeaud22a.pdf) (Borgeaud et al., ICML 2022)

<b>2021</b>

- [Surface Form Competition: Why the Highest Probability Answer Isn’t Always Right](https://arxiv.org/pdf/2104.08315.pdf) (Holtzman et al., EMNLP 2021)

    This paper investigates an very interesting problem of text scoring function used to determine a prediction $$y$$ for an input $$x$$ with LM: <b> surface form competition </b>. Specifically, given $$x$$, there could be many relevant $$y$$(s) that differ from their surface forms but share the same underlying concept in the context of $$x$$. For example, if $$x$$ is "Which is the richest country in the world", then $$y$$ could be "USA", "United States", "U.S.A" or even "U.S of A". All those answers should receive high score, however, since they come from the same finite probability mass function $$p(y \mid x)$$, they compete each other for how much probability they could get. Due to the different level of popularity of each answer $$y$$ in the training corpus, the model tends to allocate much more probability mass to popular "United States" or "USA", which consequently decrease the amount for rare "U.S of A".
    
    <b>Solution</b> Rather than calculating the ranking score $$score \; (y \mid x)$$  via $$p(y \mid x)$$ which make solutions $$y$$ compete each other, the <b>Pointwise Mutual Information (PMI)</b> is leveraged to evaluate the relevance between the input $$x$$ and the output $$y$$:

    $$score \; (y \mid x) = \text{PMI}(x, y) = log \frac{p(x,y)}{p(x) \times p(y)} = log \frac{p (x \mid y)}{p(x)}$$

    While $$p (x)$$ is constant w.r.t $$y$$ and the probability of surface form $$p (y)$$ is factored out in $$\text{PMI}(x, y)$$, the ranking of a solution $$y$$ relies solely on $$p (x \mid y)$$ that does not cause the competition between different $$y$$.

<b>2020</b>

- [Generalization through Memorization: Nearest Neighbor Language Models](https://arxiv.org/pdf/1911.00172.pdf) (Khandelwal et al., ICLR 2020):

    The paper hypothesizes that the representation learning problem may be easier than the prediction problem. For example, two sentences *Dickens is the author of* and *Dickens wrote* will essentially have the same distribution over the next word, even if they do not know what that distribution is. Given a sequence of tokens $$x = (w_1,...,w_{t-1})$$, $$k$$ nearest neighbors $$\mathcal{N}$$ of $$x$$ is retrieved from a pre-built catalog $$\mathcal{C}$$ by comparing the sentence embedding of each sequence in Eclidean space. Each nearest neighbor $$x_i$$ of $$x$$ has a next token $$y_i$$: $$(x_i, y_i) \in \mathcal{N}$$. The distribution of the next token $$y$$ of $$x$$ can be estimated via a simple linear regression: 
    $$p_{kNN} (y \mid x) = \sum_{(x_i, y_i) \in \mathcal{N}} softmax (\mathbb{1}_{y=y_i} exp (-d (\textsf{Emb}(x), \textsf{Emb}(x_i))))$$.

    The LM distribution of a token $$y$$ $$p_{LM} (y \mid x)$$ given $$x$$ is then updated by the nearest neighbor distribution $$p_{kNN} (y \mid x)$$:
    $$ p (y \mid x) = \lambda p_{kNN} (y \mid x) + (1-\lambda) p_{LM} (y \mid x)$$.

    Several advantages of nearest neighbor LM:
    - No additional training required.
    - Long-tail patterns can be explicitly memorized in the pre-built catalog $$\mathcal{C}$$ instead of encoded implicitly in model parameters. New domain can be adapted to LM by creating a new catalog for the target domain dataset.
    - $$k$$ nearest neighbor search in the embedding space of word sequences can be efficiently done using FAISS index.


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

<b>2022</b>

- [SKILL: Structured Knowledge Infusion for Large Language Models](https://aclanthology.org/2022.naacl-main.113.pdf) (Moiseev et al., NAACL 2022)

    The paper introduces <b>SKILL</b> a simple way to inject knowledge from structured data, such as a KG, into a language model, that can benefit knowledge-retrieval-based downstream tasks. <b>SKILL</b> continue to pretrain LLM directly on structured data (e.g. triples in KG) with salient-term masking without synthesizing them into equivalent natural sentences (e.g. KELM) as they found that the two approaches are competitive with each other. 

    <b>SKILL</b> demonstrates better performance than original LMs on Wikidata-related QA benchmarks as it is pre-trained on Wikidata triples. Most of the gains comes from the ability to memorize KG triples during the training. As a consequence, the model can perform very well on 1-hop questions that are supported by single triples, such as "When was Elon Musk born ?" corresponds to the triple *<Elon Musk, date of birth, ?>*. However, when it comes to answering multi-hop questions (e.g. "Who worked at the companies directed by Elon Musk ?" may correspond to two triples *<Elon Musk, owner of, ?x>* and *<?y, employer, ?x>* ) which requires not only the memorizing ability but also the reasoning ability, <b>SKILL</b> performs just slightly better than original LMs. <b> The author points out one limitation of SKILL is that the training relies on a random set of independent triples, lacking of topological structure exploitation of a KG describing how triples are connected. Addressing this issue can improve the multi-hop QA tasks</b>.

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

    Regarding the training setting, the model is trained using masked-language modelling. They found that masking salient terms instead of masking random span could significantly improve the performance on downstream tasks.

##### <b>2.4.2 Automated Knowledge Base Construction with Language Model</b>

[An overview](https://www.mpi-inf.mpg.de/fileadmin/inf/d5/teaching/ss22_akbc/8_LMs_and_KBs.pdf)

<b>Remarkable Challenges:</b>
- LM is not trained to assert factual knowledge, but to predict masked tokens/next tokens. So when it is seen predicting a true fact, is it because of the knowledge it learned or just the correlation with what it learned ([educated guess](https://aclanthology.org/2021.acl-long.146/))

- LM Probability is not a proof of veracity but rather relates to the likelihood of a token over others during the pre-training $$\rightarrow$$ LM should know its limit when answering something (e.g. chose to answer "Unknown" or "No" instead of attempting to say anything)

<b>2022</b>

- [Rewire-then-Probe: A Contrastive Recipe for Probing Biomedical Knowledge of Pre-trained Language Models](https://arxiv.org/pdf/2110.08173.pdf) (Meng et al., ACL 2022)

    <b>Contrastive-Probe for Knowledge probing from LM.</b>

    Knowledge probing approaches based on mask prediction or text generation have two typical drawbacks:

    - Multi-token span prediction: the mask prediction approaches use the MLM head to fill in a single mask token in a cloze-style query $$\rightarrow$$ if an answer entity names that contain multi-token span, the query needs to be padded with the same amount of [MASK] token.

    - The answer may not be a valid identifier of an entity: the mask prediction or text generation approaches rely on the vocabulary to generate the answer in an unconstrained way $$\rightarrow$$ the generated texts may not exist in the answer space. Furthermore, different LMs can have different vocabularies, leading to the vocabulary bias.

    The paper introduces <b>Contrastive-Probe</b> to address two above issues by avoiding using the LM head for mask prediction or text generation. Similarly to sentence embedding approaches, <b>Contrastive-Probe</b> employs the LM to encode the prompt query $$q$$(e.g. "Elvitegravir may prevent [Mask]", [Mask] can represent multiple tokens) into the embedding $$e_q$$ and encode each answer (e.g. "Epistaxis") in the complete answer space into another embeddings $$e^i_s, i=1..N$$ where $$N$$ is the size of answer space. The K-nearest neighbors $$e^k_s, k=1..K$$ of $$e_q$$ in the embedding space are considered as the answer of $$q$$. Self-supervised contrastive learning is used to rewire the pretrained LM to this answer-retrieval task. Specifically, with infoNCE objective loss, the PLM is fine-tuned on {query, answer} pairs in order for the {query, correct answer} pairs (positive samples) stay close to each other and {query, other answer in the same batch} pairs (negative samples) are pulled far apart.

    Testing on bio domain, <b>Contrastive-Probe</b> achieved several following results:
     -  <b>Contrastive-Probe</b> outperforms other probing baselines (mask prediction, text generation) ) regardless of the underlying PLM on MedLAMA benchmark for Bio domain.
     - It is effective at predict long answer (aka. multi-token span)
     - In phase with previous observation, no configuration fits all relations. Different relation require different underlying LM, different depth of tuning layer for the best performance.
     - It is pretty stable in performance where training with different dataset results small deviation and similar trend.

&nbsp;
- [Do Pre-trained Models Benefit Knowledge Graph Completion? A Reliable Evaluation and a Reasonable Approach](https://aclanthology.org/2022.findings-acl.282.pdf) (Lv et al., ACL-Findings 2022)

    The paper demonstrates that PLM-based KGC models are still left quite behind the SOTA KGC models (e.g. KGE models) because the evaluation benchmark is conducted under the closed-world assumption (CWA) where any knowledge that does not exist in a given KG is said to be incorrect. Indeed, PLM is known to implicitly contain more open knowledge unseen in a KG. By manually verify the veracity of Top-1 prediction of KGC models, they show that PLM-based models outperforms SOTA KGE-based models for the link prediction and the triple classification tasks.

    Likewise many other models, this work also make use of prompting method to elicit the knowledge from PLM. A hard prompting template is manually designed for a relation to represent the semantics of the associated triples. For example, relation *<X, member of sport teams, Y>* has the template *X plays for Y*. To further improve the expressivity of triple prompts, two other kinds of prompts are added into the triple prompt:
    - Soft prompts (i.e. learnable sentinel tokens [SP]): play as separators to signal the position of template components and entity labels in the triple prompt. For example, the prompt *X plays for Y* after adding soft prompts becomes  *[SP1] X [SP2] [SP3] plays for [SP4] [SP5] Y [SP6]*. Each relation has its own set of soft prompts $$[SP]_i, i=1..6$$ and they are all learnable via triple classification objective. 
    - Support prompts: entity definition and entity attribute are useful information that can help the KGC. Therefore, they are concatenated to the triple prompt through two templates: "[Entity]: [Entity_Definition]" and "The [Attribute] of [Entity] is [Value]". As an entity has many attributes, only few attributes are randomly selected. The results reveal that the entity definition provides more gain than entity attributes.

    Additionally, their analysis conveys two messages: <b>(i)</b> by counting number of sentences in the training that contain both the head and the tail of a triple, it indicates that PLM-based KGC still outperforms KGE-based KGC on the triples with zero co-occurrence of {head, tail} in the training set $$\rightarrow$$ they argue PLMs, apart from seeing many facts in the massive text, have the ability to reason the knowledge. <b>(i)</b> PLM-based KGC models are less sensitive to the size of training dataset where reducing the training data size decreases slightly the prediction accuracy.
   
- [SimKGC: Simple Contrastive Knowledge Graph Completion with Pre-trained Language Models](https://arxiv.org/pdf/2203.02167.pdf) (Wang et al. ACL 2022)

    <b>SimKGC: Promptless method for KGC based on sentence embedding</b>

    To predict an entity $$e_i \in KG \; \mathcal{E}$$ for a triple $$<h, r, ?>$$, SimKGC employs a PLM-based bi-encoder architecture where two encoders do not share parameters. One encoder computes the relation-aware embedding $$e_{hr}$$ for the head entity $$h$$ from the concatenation of the descriptions of the head entity and the relation: "[header_description] [SEP] [relation_description]". Another encoder is leveraged to compute the embedding of the description of the candidate tail entity $$e_t$$. Candidate tail entities $$e^i_t$$ are ranked according to the cosine similarity between its embedding and the relation-aware embedding of the head entity $$e_{hr}$$. The bi-encoder is trained to learn useful representation for head entity and tail entity in the triple using contrastive learning.

    The paper argues that the reason why previous contrastive learning-based models are lag behind SOTA KGE-based models highly involves the ineffectiveness of training setting for contrastive learning where they use small negative sample size ($$\approx$$ 1..5 due to computational complexity) and the margin loss. Indeed, by augmenting the number of negative sample per positive sample (e.g. 256) and changing the margin loss to InfoNCE loss, they obtain much better performance and outperform KGE-based models. 

    For further improvement, in addition to in-batch negative, SimKGC also combine two other strategies for generating negative samples:
    - Pre-batch Negatives: sample batches at training step $$t-1$$, $$t-2$$... can be considered as negative samples for current training batch at step $$t$$.
    - Self-Negatives: triple $$<h, r, h>$$ (tail entity is predicted as head entity) is seen as a hard negative sample for the triple $$<h, r, ?>$$ $$\rightarrow$$ this makes the model rely less on the spurious text matching/overlapping to make the prediction.

    Lastly, the work also stresses that predicting *one-to-many, many-to-one, many-to-many* relations is more difficult.


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

#### <b>2.5. Domain Adaptation of Language Model </b>

<b>2022</b>

- [The Trade-offs of Domain Adaptation for Neural Language Models](https://aclanthology.org/2022.acl-long.264.pdf) (Grangier et al., ACL 2022)

    This paper provides some evidences using concepts of machine learning theory to support past prevailing empirical practices/observations for domain adaption of LM.

    <b>1. In-domain training</b>

    The loss of fitting a LM  to a domain $$\mathcal{D}$$ is decomposed as 3 components: 
    $$\mathcal{L} (\theta_D, \mathcal{D}) = \mathcal{L}_H(\mathcal{D}) + \mathcal{L}_{app} (\mathcal{D}, \Theta) + \mathcal{L}_{est} (\mathcal{D}, \Theta, D)$$
    where 
    - $$\mathcal{L}_H(\mathcal{D})$$ is the intrinsic uncertainty of the domain $$\mathcal{D}$$ itself.
    - $$\mathcal{L}_{app} (\mathcal{D}, \Theta)$$ is the approximation error of using a LM parameterized by $$\theta \in \Theta$$ to approximate the true distribution $$P( \bullet \mid \mathcal{D})$$ over domain $$\mathcal{D}$$: $$\mathcal{L}_{app} (\mathcal{D}, \Theta) =  \min_{\theta \in \Theta} \mathcal{L}(\theta; \mathcal{D}) - H(P( \bullet \mid \mathcal{D}))$$ where $$\mathcal{L}(\theta; \mathcal{D})$$ is the expectation of risk of using LM $$P(\bullet \mid \theta)$$ to approximate true distribution $$P(\bullet \mid \mathcal{D})$$:  $$\mathcal{L}(\theta; \mathcal{D}) = -\sum_{y \in \mathcal{D}} log \; P(y \mid \theta) P(y \mid \mathcal{D})$$. Larger model with deeper, wider layers has more capacity, consequently, can reduce this error.
    - $$\mathcal{L}_{est} (\mathcal{D}, \Theta, D)$$ is the error of using the LM parameters empirically estimated from a subset $$D \subset \mathcal{D}$$ to represent the true distribution $$P(\bullet \mid \mathcal{D})$$: $$\mathcal{L}_{est} (\mathcal{D}, \Theta, D) = \mathcal{L} (\theta_D, \mathcal{D}) - \min_{\theta} \mathcal{L}(\theta; \mathcal{D})$$ where $$\theta_D = arg \min_{\theta \in \Theta} \mathcal{L} (\theta, D) $$.

    For a given training set size, increasing the size of the model can decrease $$\mathcal{L}_{app} (\mathcal{D}, \Theta)$$ but can increase $$\mathcal{L}_{est} (\mathcal{D}, \Theta, D)$$ due to overfiting $$\rightarrow$$ $$\mathcal{L}_{app} \; vs. \; \mathcal{L}_{est}$$ trade-off or VC-dimension trade-off.


    <b>2. Out-of-domain training</b>

    Given two LMs pretrained on two generic domaines $$\mathcal{D}$$ and $$\mathcal{D'}$$, which one we should choose to adapt it to a target domain $$\mathcal{T}$$ ?. Intuitively, we choose the one whose distribution is closer to the target distribution $$\mathcal{T}$$ or KL divergence between two distributions is smaller as the generalization loss of adapting LM parameters $$\theta_D$$ estimated from generic domain $$\mathcal{D}$$ for $$\mathcal{T}$$ is upper-bounded by the KL divergence $$KL(\mathcal{D}, \mathcal{T})$$

    $$\forall \epsilon, \exists D \subset \mathcal{D}, \mathcal{L}(\theta_D; \mathcal{T}) \leqslant H(\mathcal{T}) + KL(\mathcal{D}, \mathcal{T}) + \epsilon $$

    <b>3. Fine-Tuning & Multitask Learning</b>

    Pre-training a LM on a large out-of-domain corpus $$D$$ then fine-tuning it on a small in-domain corpus $$T$$ implicitly involves the trade-off between empirical losses over $$T$$ and $$D$$. This trade-off is controlled by the number of fine-tuning steps $$n_{ft}$$: $$ \parallel \theta_{ft} - \theta_D  \parallel_2 \;\leqslant \lambda n_{ft} g_{max} $$ where $$\lambda$$ is maximum learning rate, $$g_{max}$$ is upper bound of update norm. More fine-tuning steps $$n_{ft}$$, larger possible distance between $$\theta_{ft}$$ and $$\theta_D$$ is, meaning that $$\theta_{ft}$$ could be no longer optimal for $$D$$ where $$\mathcal{L}(\theta_{ft}; D)$$ may be far from the optimum $$\mathcal{L}(\theta_{D}; D)$$. For this reason, fine-tuning is also considered as a regularization technique.
    

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
# sentence-embeddings
This repository provides an easy way to create document/sentence embeddings for other NLP tasks downstream.  

There are three primary methods for creating document/sentence embeddings:
1. [word2vec-sif](https://github.com/oborchers/Fast_Sentence_Embeddings)
2. [doc2vec](https://radimrehurek.com/gensim/models/doc2vec.html)
3. [pre-trained transformer models](https://github.com/UKPLab/sentence-transformers)

# Installation

First, clone the repository. 

In your terminal, run:  
```
pip install -r requirements.txt
```

# Basic Usage

```python
>>> import embeddings
>>> text_embeddings = embeddings.word2vec_sif_embeddings(texts, n_dimensions = 300, model=None)
```

Where `texts` is a list of sentences or documents in the form of strings.

# Resources/References
* [doc2vec paper](https://arxiv.org/pdf/1405.4053v2.pdf)
* [word2vec-sif paper](https://openreview.net/pdf?id=SyK00v5xx)
* [word vector pre-trained models for word2vec-sif sentence embeddings](https://github.com/RaRe-Technologies/gensim-data)
* [sentence-transformer pre-trained models](https://docs.google.com/spreadsheets/d/14QplCdTCDwEmTqrn1LH4yrbKvdogK4oQvYO1K1aPR5M/edit#gid=0)
* [sentence-transformers documentation](https://www.sbert.net/index.html)
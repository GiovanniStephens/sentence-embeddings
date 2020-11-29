def word2vec_sif_embeddings(texts, n_dimensions = 300,  model_name = "glove-wiki-gigaword-300"):
    """
    This model uses word2vec or other pretrained word embeddings to create sentence embeddings
    using the smooth inverse frequency (SIF) method. The method basically calculates a weighted
    embedding for the whole sentence by weighting each word as a/(a + p(w)) where a is a hyperparameter
    and p(w) is the estimated probability of the word appearing in the corpus. Once that has been done, 
    subtract the first principal component from the vectors. This gives really quick embeddings that 
    perform really well. 
    
    Original paper: "A Simple but Tough-to-Beat Baseline for Sentence Embeddings"

    I would use this model without pretrained word vectors if there was a lot of custom
    vocabulary (e.g. business specific vocabulary). 
    Another use case would be to use pretrained word vectors to get really fast sentence 
    embeddings on a lot of data that does not have a lot of vocab.

    :texts: list or list-like object with sentence/paragraph strings.
    :n_dimensions: Number of dimensions for the embedding to be.
    :model_name: name of the pretrained model to use.

    :return: list of sentence embeddings. 
    """
    import gensim.downloader as api
    import gensim
    from gensim.models import Word2Vec
    from fse.models import sif
    from fse import IndexedList

    if model_name == None or model_name == '':
        corpus = [utterance.lower().split() for utterance in texts]
        model = Word2Vec(corpus, size = n_dimensions, min_count=1, workers=-1)
    else:
        model = api.load(model_name)
        corpus = [[word for word in utterance.lower().split() if word in model.vocab] for utterance in texts]

    # Get document vectors
    se = sif.SIF(model)
    s = IndexedList(corpus)
    se.train(s)
    return se.sv.vectors


def doc2vec_embeddings(texts, n_dimensions = 300):
    """
    This model creates paragraph embeddings by creating another vector when training
    the word vectors. In a way, it's like a weighted paragraph vector based on the 
    word vectors. 

    original paper: "Distributed Representations of Sentences and Documents"
        https://arxiv.org/pdf/1405.4053v2.pdf 

    :texts: list or list-like object with sentence/paragraph strings.
    :n_dimensions: Number of dimensions for the embedding to be.

    :return: list of doc2vec sentence embeddings. 
    """
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]
    model = Doc2Vec(documents, vector_size=n_dimensions, window=2, min_count=1, workers=-1)
    return [model.docvecs[i] for i in range(len(texts))]


def pretrained_transformer_embeddings(texts, model_name = 'distilbert-base-nli-stsb-mean-tokens'):
    """
    Returns pretrained sentence embeddings using transformer models. 
    List of models can be found here:
    https://docs.google.com/spreadsheets/d/14QplCdTCDwEmTqrn1LH4yrbKvdogK4oQvYO1K1aPR5M/edit#gid=0

    :texts: list or list-like object with sentence/paragraph strings.
    :model_name: name of the pretrained model to use.

    :return: list of sentence embeddings. 
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(texts)

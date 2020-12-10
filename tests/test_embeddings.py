import unittest

class test_embeddings(unittest.TestCase):

    def _get_module(self, module_name):
        import importlib
        return importlib.import_module(module_name)

    def _cosine_similarity(self, v1,v2):
        "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(v1)):
            x = v1[i]; y = v2[i]
            sumxx += x*x
            sumyy += y*y
            sumxy += x*y
        return sumxy/(sumxx*sumyy)**0.5

    def test_cosine_same(self):
        """Tests my cosine similarity helper function."""
        self.assertEqual(self._cosine_similarity([1,1,1], [1,1,1]), 1)

    def test_cosine_different(self):
        """Tests cosine similarity helper function with different vectors""" 
        self.assertLess(self._cosine_similarity([1,1,1], [1,1,2]), 1)

    def test_word2vec_sif_embeddings_glove(self):
        """Tests using pre-trained glove word embeddings to 
        create sentence embeddings with -sif weightings.
        The similarity between utterances 1 and 2 should be more than 
        1 and 3. """
        embed = self._get_module('embeddings')
        test_utterances = [
            'Hi, this phrase is similar to the second phrase.'
            , 'Hi, that phrase is like this phrase.'
            , 'I have nothing to do with the others.'
        ]
        embeddings = embed.word2vec_sif_embeddings(test_utterances)
        sim_1 = self._cosine_similarity(embeddings[0], embeddings[1])
        sim_2 = self._cosine_similarity(embeddings[0], embeddings[2])
        self.assertGreater(sim_1, sim_2)

    def test_word2vec_sif_embeddings_w2v(self):
        """Tests using freshly trained word2vec word embeddings to 
        create sentence embeddings with -sif weightings.
        
        The similarity between utterances 1 and 2 should be more than 
        1 and 3. """
        embed = self._get_module('embeddings')
        test_utterances = [
            'Hi, this phrase is similar to the second phrase.'
            , 'Hi, that phrase is like this phrase.'
            , 'I have nothing to do with the others.'
        ]
        embeddings = embed.word2vec_sif_embeddings(test_utterances, model_name=None)
        sim_1 = self._cosine_similarity(embeddings[0], embeddings[1])
        sim_2 = self._cosine_similarity(embeddings[0], embeddings[2])
        self.assertGreater(sim_1, sim_2)

    def test_doc2vec_embeddings(self):
        """Tests doc2vec embeddings. It seems to work better when there is more
        data.
        The similarity between utterances 1 and 2 should be more than 
        1 and 3."""
        embed = self._get_module('embeddings')
        test_utterances = [
            'Hi, this phrase is similar to the second phrase.'
            , 'Hi, this phrase is like to the second phrase.'
            , 'I have nothing to do with the others.'
            , 'I have nothing to do with the others 2.'
            , 'Do I need more data in here to get better embeddings?'
        ]
        embeddings = embed.doc2vec_embeddings(test_utterances)
        sim_1 = self._cosine_similarity(embeddings[0], embeddings[1])
        sim_2 = self._cosine_similarity(embeddings[0], embeddings[2])
        self.assertGreater(sim_1, sim_2, """It's likely that this failed due to 
        how the initial neural network weights were initialised (at random). As a result,
        because there is not much data, results for this are a bit hit or miss. """)

    def test_doc2vec_embeddings_same(self):
        """Tests doc2vec embeddings exact match vs not exact. 
        It seems to work better when there is more data.
        The similarity between utterances 1 and 2 should be more than 
        1 and 3."""
        embed = self._get_module('embeddings')
        test_utterances = [
            'Hi, this phrase is similar to the second phrase.'
            , 'Hi, this phrase is like to the second phrase.'
            , 'I have nothing to do with the others.'
            , 'I have nothing to do with the others.'
            , 'Do I need more data in here to get better embeddings?'
        ]
        embeddings = embed.doc2vec_embeddings(test_utterances)
        sim_1 = self._cosine_similarity(embeddings[0], embeddings[1])
        sim_2 = self._cosine_similarity(embeddings[2], embeddings[3])
        self.assertLess(sim_1, sim_2, """It's likely that this failed due to 
        how the initial neural network weights were initialised (at random). As a result,
        because there is not much data, results for this are a bit hit or miss. """)

    def test_pretrained_transformer_embeddings(self):
        """Tests pre-trained transformer embeddings. 
        The similarity between utterances 1 and 2 should be more than 
        1 and 3."""
        embed = self._get_module('embeddings')
        test_utterances = [
            'Hi, this phrase is similar to the second phrase.'
            , 'Hi, that phrase is like this phrase.'
            , 'I have nothing to do with the others.'
        ]
        embeddings = embed.pretrained_transformer_embeddings(test_utterances)
        sim_1 = self._cosine_similarity(embeddings[0], embeddings[1])
        sim_2 = self._cosine_similarity(embeddings[0], embeddings[2])
        self.assertGreater(sim_1, sim_2)

if __name__ == '__main__':
    unittest.main()
import os
import pickle
import math
import re

from .index import InvertedIndexReader
from .compression import VBEPostings
from mpstemmer import MPStemmer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from operator import itemgetter


class BSBIIndex:
    def __init__(self, data_dir='collections', output_dir='index', postings_encoding=VBEPostings, index_name="main_index"):
        self.term_id_map = dict()
        self.term_counter = 0
        self.doc_id_map = dict()
        self.doc_counter = 0
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding

        self.stemmer = MPStemmer(check_nonstandard=False)
        self.remover = StopWordRemoverFactory().create_stop_word_remover()

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""

        with open(os.path.join('search', self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join('search', self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

    def pre_processing_text(self, content):
        """
        Melakukan preprocessing pada text, yakni stemming dan removing stopwords
        """
        # https://github.com/ariaghora/mpstemmer/tree/master/mpstemmer

        stemmed = self.stemmer.stem(content)
        return self.remover.remove(stemmed)

    def retrieve_tfidf(self, query, k=10):
        self.load()
        scores = {}

        query_tokens = re.findall(r'\w+', query)
        query_tokens = [self.pre_processing_text(token) for token in query_tokens]
        with InvertedIndexReader(self.index_name, self.postings_encoding, self.output_dir) as indices:
            postings_dict = indices.postings_dict
            N = len(indices.doc_length)

            for token in query_tokens:
                if token == "" or token not in self.term_id_map:
                    continue

                term_id = self.term_id_map[token]

                postings_list, tf_list = indices.get_postings_list(term_id)
                postings_tf = list(zip(postings_list, tf_list))

                df_t = postings_dict[term_id][1]
                idf = math.log10(N/df_t)

                for doc_id, tf in postings_tf:
                    if doc_id not in scores:
                        scores[doc_id] = 0
                    score = 1 + math.log10(tf)
                    score = score * idf
                    scores[doc_id] += score

        sorted_k = sorted(scores.items(), key=itemgetter(1), reverse=True)[:k]
        return [(score, self.doc_id_map[doc_id]) for doc_id, score in sorted_k]

    def retrieve_bm25(self, query, k=10, k1=1.2, b=0.75):
        self.load()
        scores = {}

        query_tokens = re.findall(r'\w+', query)
        query_tokens = [self.pre_processing_text(token) for token in query_tokens]

        with InvertedIndexReader(self.index_name, self.postings_encoding, os.path.join('search', self.output_dir)) as indices:
            postings_dict = indices.postings_dict
            N = len(indices.doc_length)
            avg_dl = indices.avg_doc_length

            for token in query_tokens:
                if token == "" or token not in self.term_id_map:
                    continue

                term_id = self.term_id_map[token]

                postings_list, tf_list = indices.get_postings_list(term_id)
                postings_tf = list(zip(postings_list, tf_list))

                df_t = postings_dict[term_id][1]
                idf = math.log10(N / df_t)

                for doc_id, tf in postings_tf:
                    if doc_id not in scores:
                        scores[doc_id] = 0
                    score = (k1 + 1) * tf
                    score /= k1 * ((1 - b) + b * indices.doc_length[doc_id] / avg_dl) + tf
                    score = score * idf
                    scores[doc_id] += score

        sorted_k = sorted(scores.items(), key=itemgetter(1), reverse=True)[:k]
        return [(score, self.doc_id_map[doc_id]) for doc_id, score in sorted_k]

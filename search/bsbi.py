import os
import pickle
import contextlib
import heapq
import math
import ir_datasets
import nltk

from .index import InvertedIndexReader
from .compression import VBEPostings
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from operator import itemgetter

nltk.download('punkt')
nltk.download('stopwords')


class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """

    def __init__(self, output_dir="index", postings_encoding=VBEPostings, index_name="main_index"):
        self.term_id_map = dict()
        self.id_term_map = []
        self.doc_id_map = dict()
        self.id_doc_map = []
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""

        with open(os.path.join('search', self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map, self.id_term_map = pickle.load(f)
        with open(os.path.join('search', self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map, self.id_doc_map = pickle.load(f)

    def pre_processing_text(self, content):
        """
        Melakukan preprocessing pada text, yakni stemming dan removing stopwords
        """

        tokens = word_tokenize(content)
        tokens = [token for token in tokens if token not in self.stop_words]
        tokens = [self.stemmer.stem(token) for token in tokens]

        return tokens

    def retrieve_tfidf(self, query, k=10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan:
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        # TODO
        self.load()
        scores = {}

        query_tokens = self.pre_processing_text(query)
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
        return [(score, self.id_doc_map[doc_id]) for doc_id, score in sorted_k]

    def retrieve_bm25(self, query, k=10, k1=1.2, b=0.75):
        """
        Melakukan Ranked Retrieval dengan skema scoring BM25 dan framework TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        """
        # TODO
        self.load()
        scores = {}

        query_tokens = self.pre_processing_text(query)

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
        return [(score, self.id_doc_map[doc_id]) for doc_id, score in sorted_k]

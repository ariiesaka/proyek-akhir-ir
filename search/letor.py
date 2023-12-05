import os.path

import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from gensim.models import FastText
from mpstemmer import MPStemmer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import re
import lightgbm
from .constants import OS_SEP_WIN

class Letor:
    def __init__(self) -> None:

        # copy semua file di https://drive.google.com/drive/folders/17VarHWduvxCHS2k-TgAXLCvL-nCcLpE6?hl=id
        # ke folder ./letor-model
        self.encoder = FastText.load(os.path.join('search', 'letor-model', 'fasttext_model'))
        self.model = lightgbm.Booster(model_file=os.path.join('search', 'letor-model', 'letor_fasttext12.txt'))
        self.stemmer = MPStemmer(check_nonstandard=False)
        self.remover = StopWordRemoverFactory().create_stop_word_remover()

    def get_cosine(self, doc_embed, query_embeds):
        result = 0
        for query_embed in query_embeds:
            result += cosine(doc_embed, query_embed/np.linalg.norm(query_embed))
        return result/len(query_embeds)

    def jaccard(self, query, doc):
        q_set = set(self.preprocess_text(query.lower()))
        d_set = set(self.preprocess_text(doc.lower()))
        return len(q_set & d_set) / len(q_set | d_set)

    def encode(self, texrOrTokens, merged=False):
        tokens = texrOrTokens
        if type(texrOrTokens) == str:
            tokens = self.preprocess_text(texrOrTokens.lower())
        if merged:
            mean_vector = self.encoder.wv.get_mean_vector(tokens) 
            return mean_vector/np.linalg.norm(mean_vector)
        else:
            return [self.encoder.wv[token] for token in tokens]

    def preprocess_text(self, text:str):
        tokens = re.findall(r'\w+', text)
        valid_tokens = []
        for token in tokens:
            stemmed = self.stemmer.stem(token)
            removed_stop_word = self.remover.remove(stemmed)
            if removed_stop_word != '':
                valid_tokens.append(removed_stop_word)
        return valid_tokens
    
    def predict(self, query, doc_score_names):
        if len(doc_score_names) == 0:
            return []
        
        docs = []
        bm25s = []
        for doc_score, doc_id in doc_score_names:
            path_parts = doc_id.split(OS_SEP_WIN)
            with open(os.path.join('search', 'collections', path_parts[-2], path_parts[-1]), 'r', encoding='utf8') as doc_file:
                doc = doc_file.read()
                docs.append(doc)
                bm25s.append(doc_score)
        
        df = pd.DataFrame({"document": docs, "bm25": bm25s})
        df["query"] = df.document.apply(lambda x:query)
        X = self.generate_features(df)
        letor_scores = self.model.predict(X)
        did_scores = [(letor_score, did, content) for letor_score, (_, did), content in zip(letor_scores, doc_score_names, docs)]
        did_scores.sort(key=lambda did_score: did_score[0], reverse=True)
        return did_scores

    def generate_features(self, df:pd.DataFrame):
        new_df = df[["query", "document", "bm25"]].copy()
        new_df["query_embed"] = new_df["query"].apply(lambda query: self.encode(query))
        new_df["doc_embed"] = new_df["document"].apply(lambda doc: self.encode(doc, merged=True))
        new_df["cosine"] = new_df.apply(lambda row: self.get_cosine(row["doc_embed"], row["query_embed"]), axis=1)
        new_df["jaccard"] = new_df.apply(lambda row: self.jaccard(row["query"], row["document"]), axis=1)
        new_df["exact_match"] = new_df.apply(lambda row: 1 if row["query"].lower() in row["document"].lower() else 0, axis=1)
        new_df["wm_dist"] = new_df.apply(lambda row: self.encoder.wv.wmdistance(row["query"].lower(), row["document"].lower()), axis=1)
        new_df["doc_len"] = new_df["document"].apply(len)
        new_df["query_len"] = new_df["query"].apply(len)
        new_df["query_embed"] = new_df["query"].apply(lambda query: self.encode(query, merged=True))

        return np.concatenate((
            np.stack(new_df["query_embed"].to_numpy(), axis=0),
            np.stack(new_df["doc_embed"].to_numpy(), axis=0),
            new_df[['bm25', 'cosine', 'jaccard', 'exact_match', 'wm_dist', 'doc_len', 'query_len']].to_numpy()
        ), axis=1)
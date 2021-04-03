import xgboost as xgb
import numpy as np
import pandas as pd
import logger
import tqdm
import nltk
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import sigmoid_kernel
from jiwer import wer

class XGBoost(xgb.XGBRegressor):
    def __init__(self, max_depth):
        super().__init__(max_depth=max_depth, learning_rate=0.1, silent=True, objective='reg:squarederror', nthread=4)


def sentence_pair_features(sentence1, sentence2):
    def lemmatize(sentence, ignore_stop_words=False):
        pos_map = {
            'J': nltk.corpus.wordnet.ADJ,
            'v': nltk.corpus.wordnet.VERB,
            'N': nltk.corpus.wordnet.NOUN,
            'R': nltk.corpus.wordnet.ADV
        }

        seq = nltk.tokenize.word_tokenize(sentence)
        seq_tokens_pos = nltk.pos_tag(seq)
        lemmatizer = nltk.stem.WordNetLemmatizer()
        if ignore_stop_words:
            seq = [word for word in seq if word not in nltk.corpus.stopwords.words('english')]
            seq_tokens_pos = [token_pos for token_pos in seq_tokens_pos if token_pos[0] in seq]
        seq = [lemmatizer.lemmatize(w[0], pos_map.get(w[1][0], nltk.corpus.wordnet.NOUN)) for w in seq_tokens_pos]
        return seq

    def levenshtein_distance(sentence1, sentence2):
        seq1 = lemmatize(sentence1, ignore_stop_words=True)
        seq2 = lemmatize(sentence2, ignore_stop_words=True)
        return nltk.edit_distance(seq1, seq2, transpositions=False) / max(len(seq1), len(seq2))

    def sentances_ngo(seq1, seq2, n, level='c'):
        def ngo(list1, list2):
            intersection = list(set(list1).intersection(list2))
            n_inter = len(intersection)
            n_list1 = len(list1)
            n_list2 = len(list2)
            if n_inter == 0:
                return 0
            result = 2 / (n_list1 / n_inter + n_list2 / n_inter)
            return result

        if level == 'w':
            seq1 = nltk.tokenize.word_tokenize(seq1)
            seq2 = nltk.tokenize.word_tokenize(seq2)

        if level == 'l':
            seq1 = lemmatize(seq1)
            seq2 = lemmatize(seq2)

        grams_1 = list(nltk.ngrams(seq1, n))
        grams_2 = list(nltk.ngrams(seq2, n))
        return ngo(grams_1, grams_2)

    n_char_level = [3, 4, 5]
    n_word_level = [2, 3]
    modes = {
        'c': n_char_level,
        'w': n_word_level,
        'l': n_word_level
    }
    features = {}

    for k, v in modes:
        for n in v:
            features[f'ngo_{k}-{n}'] = sentances_ngo(sentence1, sentence2, n, level=k)

    features['lev'] = levenshtein_distance(sentence1, sentence2)
    features['wer'] = wer(sentence1, sentence2)
    features['bleu'] = nltk.translate.bleu(sentence1, sentence2)

    return features

def all_sentence_pair_features(sent1_batch, sent2_batch):
    features = []
    for i in tqdm(range(len(sent1_batch))):
        features.append(sentence_pair_features(sent1_batch[i], sent2_batch[i]))

    return pd.DataFrame(features)

def compute_bow_features(df, tfidf_bow):
    sent1_bow = list(tfidf_bow.transform(list(df.sent1)))
    sent2_bow = list(tfidf_bow.transform(list(df.sent2)))

    kernels = {
        'cosine': lambda x, y: cosine(x.T.toarray(), y.T.toaray()),
        'pearson': lambda x, y: pearsonr(x.T.toarray(), y.T.toaray()),
        'sigmoid': sigmoid_kernel
    }

    features = []
    for i in tqdm(range(len(sent1_bow))):
        scores = {}
        for k, v in kernels:
            scores[k] = v(sent1_bow[i], sent2_bow[i])
        features.append(scores)

    return pd.DataFrame(features)

def check_nan_in_features(features_df, ind=None):
    check_nan = np.isnan(features_df).any(axis=1)
    if check_nan.any():
        idx = features_df[check_nan].index
        idx = idx if ind is None else [ind[i] for i in idx]
        logger.debug(f'Got NaN for these index{idx}')
        features_df.fillna(0, inplace=True)

def compute_features(pairs, headers, tf_idf, sentence_field, df_loader, dts_loader):
    pair_features, bow_features, wef_features = None, None, None

    df = df_loader(pairs, headers)

    pair_features = all_sentence_pair_features(df.sent1.values, df.sent2.values)
    check_nan_in_features(pair_features)

    tfidf_bow, _ = tf_idf
    bow_features = compute_bow_features(df, tfidf_bow)
    check_nan_in_features(bow_features)

    return pair_features, bow_features, wef_features

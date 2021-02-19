from .perceptron import Perceptron
from .xgboost import XGBoost
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import average
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer

class Assistant:
    def __init__(self, train_params, corpus, scores):
        self.train_params = train_params

        self.tf_idf = self.make_tfidf(corpus)
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(corpus)
        sequences = tokenizer.texts_to_sequences(corpus)
        maxlen = max([len(x) for x in sequences])
        padded_data = pad_sequences(sequences, padding='post', truncating='post', maxlen=maxlen)

        inputs = Input(shape=(maxlen,))        
        self.ensemble_model = self.make_ensemble(inputs, train_params['model_params'])
        self.ensemble_model.compile()
        self.ensemble_model.fit(padded_data, scores)

    def make_ensemble(self, inputs, models_params) -> Model:
        models_dict = {
            'perceptron': Perceptron,
            'xgboost': XGBoost
        }

        model_list = []
        for k, v in models_params.items():
            model_list.append(models_dict[k](v))
        outputs = average(model_list)
        return Model(input=inputs, output=outputs)

    def make_tfidf(self, doc):
        tf = TfidfVectorizer(lowercase=False)
        tf.fit(doc)
        return tf, dict(zip(tf.get_feature_names(), tf.idf_))

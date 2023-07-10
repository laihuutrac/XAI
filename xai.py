import argparse
from pathlib import Path
from typing import List, Any
import spacy
#
import numpy as np
from lime.lime_text import LimeTextExplainer
import lime.lime_image
import sklearn.pipeline
from sklearn.preprocessing import LabelEncoder
# expaliner shap
import shap
# Text preprocessing function
import re
import string
str_punc = string.punctuation.replace(',', '').replace("'",'')

def clean(text):
    global str_punc
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = text.lower()
    return text   

def explainer_class(method: str, filename: str) -> Any:
    "Instantiate class using its string name"
    classname = METHODS[method]['class']
    class_ = globals()[classname]
    return class_(filename)

def tokenizer(text: str) -> str:
    "Tokenize input string using a spaCy pipeline"
    nlp = spacy.blank('en')
    # nlp.add_pipe(nlp.create_pipe('sentencizer'))  # Very basic NLP pipeline in spaCy
    nlp.add_pipe('sentencizer')
    doc = nlp(text)
    tokenized_text = ' '.join(token.text for token in doc)
    return tokenized_text

METHODS = {
    'logistic': {
        'class': "LogisticExplainer",
        'file': "model/LogisticRegression/logisticRegression.pkl",
        'name': "Logistic Regression",
        'lowercase': False,
    },
    'svm': {
        'class': "SVMExplainer",
        'file': "model/SupportVectorMachine/supportVectorMachine.pkl",
        'name': "Support Vector Machine",
        'lowercase': False,
    },
    'bilstm': {
        'class': "BiLstmExplainer",
        'file': "model/BiLstm",
        'name': "BiLstm",
        'lowercase': True,
    },
    'cnn-lstm': {
        'class': "CnnLstmExplainer",
        'file': "model/CNN-Lstm",
        'name': "CNN-Lstm",
        'lowercase': True,
    },

}

class LogisticExplainer:
    """Class to explain classification results of a scikit-learn
       Logistic Regression Pipeline. The model is trained within this class.
    """
    def __init__(self,path_to_model: str) ->None:

        import pickle
        self.classifier= pickle.load(open(path_to_model,'rb'))

    def predict(self, texts: List[str])->np.array([float, ...]):
        """Generate an array of predicted scores (probabilities) from sklearn
        Logistic Regression Pipeline."""
        # classifier=sefl.train()
        probs=self.classifier.predict_proba(texts)
        return probs
    
class SVMExplainer:
    """Class to explain classification results of a scikit-learn linear Support Vector Machine
       (SVM) Pipeline. The model is trained within this class.
    """
    def __init__(self,path_to_model: str) ->None:

        import pickle
        self.classifier= pickle.load(open(path_to_model,'rb'))
    
    def predict(self, texts: List[str])->np.array([float, ...]):
        """Generate an array of predicted scores (probabilities) from sklearn
        Support Vectors Machine Pipeline."""
       
        probs= self.classifier.predict_proba(texts)
        return probs  
    
class BiLstmExplainer:
    """Class to explain classification results of BiLSTM.
       Assumes that we already have a trained BiLSTM model with which to make predictions.
    """
    def __init__(self,path_to_model: str) ->None:
        from keras.models import load_model
        self.classifier = load_model(path_to_model, compile=False)
    def predict(self,text):
        from keras_preprocessing.sequence import pad_sequences
        import pickle
        with open('model/tokenizer.pickle', 'rb') as handle:
            tokenizer_lstm = pickle.load(handle)
        _seq = tokenizer_lstm.texts_to_sequences(text)
        _text_data = pad_sequences(_seq, maxlen=256)
        return self.classifier.predict(_text_data)
    
class CnnLstmExplainer:
    """Class to explain classification results of CNN-LSTM.
        Assumes that we already have a trained CNN-LSTM model with which to make predictions.
    """
    def __init__(self,path_to_model: str) ->None:
        from keras.models import load_model
        self.classifier = load_model(path_to_model, compile=False)
    def predict(self,text):
        from keras_preprocessing.sequence import pad_sequences
        import pickle
        with open('model/tokenizer.pickle', 'rb') as handle:
            tokenizer_lstm = pickle.load(handle)
        _seq = tokenizer_lstm.texts_to_sequences(text)
        _text_data = pad_sequences(_seq, maxlen=256)
        return self.classifier.predict(_text_data)
    
def limeExplainer(method: str,
              path_to_file: str,
              text: str,
              lowercase: bool,
              num_samples: int) -> LimeTextExplainer:
    """Run LIME explainer on provided classifier"""

    model = explainer_class(method, path_to_file)
    predictor = model.predict
    # Lower case the input text if requested (for certain classifiers)
    if lowercase:
        text = text.lower()

    # Create a LimeTextExplainer
    explainer = LimeTextExplainer(
        # Specify split option
        split_expression=lambda x: x.split(),
        # Our classifer uses trigrams or contextual ordering to classify text
        # Hence, order matters, and we cannot use bag of words.
        bow=False,
        # Specify class names for this case
        class_names=[0, 1, 2, 3, 4, 5]
    )

    # Make a prediction and explain it:
    exp = explainer.explain_instance(
        text,
        classifier_fn=predictor,
        top_labels=1,
        num_features=20,
        num_samples=num_samples,
    )
    return exp

def shapExplainer(method: str,
                  path_to_file: str,
                  text: str,
                  lowercase: bool,
                  )->Any:
    
    model = explainer_class(method, path_to_file)
    predictor = model.predict
    masker = shap.maskers.Text(tokenizer=r"\W+")
    exp = shap.Explainer(predictor,masker=masker,output_names=[0,1,2,3,4,5,])
    
    shapValues=exp([text])
    return shapValues


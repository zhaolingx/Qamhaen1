import os
import json
from nltk.tokenize import word_tokenize
from gensim.utils import tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import *
import keras
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix
import xgboost as xgb
from tensorflow import keras
from keras.layers import LSTM, Dense, Dropout, RNN
from keras.models import Sequential, load_model
import tensorflow as tf
from keras.layers import Dense, Embedding, Dropout, Input, Concatenate
# from experiment.deep_learning import *
from experiment.model import *
from scipy.spatial import distance as dis
import distance._levenshtein
from representation.word2vec import Word2vector
dirname = os.path.dirname(__file__)
# from experiment.model import Transformer
# from experiment.mask import  *



embedding_method = 'bert'
dataset = pickle.load(open(os.path.join(dirname, '../data/bugreport_patch_array_' + embedding_method + '.pickle'), 'rb'))
bugreport_vector = np.array(dataset[0]).reshape((len(dataset[0]), -1))
commit_vector = np.array(dataset[1]).reshape((len(dataset[1]), -1))
labels = np.array(dataset[2])

# combine bug report and commit message of patch
train_features = np.concatenate((bugreport_vector, commit_vector), axis=1)
# standard data
scaler = StandardScaler().fit(train_features)
x_train = scaler.transform(train_features)
y_train = labels

# qa_attention QUATRAIN
seq_maxlen = 64
y_train = np.array(y_train).astype(float)
x=y_train.shape
print(x)
x_train_q = x_train[:, :1024]
x_train_a = x_train[:, 1024:]
x_train_q = np.reshape(x_train_q, (x_train_q.shape[0], seq_maxlen, -1))
x_train_a = np.reshape(x_train_a, (x_train_a.shape[0], seq_maxlen, -1))
print(x_train_q.shape[0])
# quatrain_model = Transformer(num_layers=12, d_model=240, heads=3, d_ff=128,
#                   target_vocab_size=12, dropout=0.01)
# x_train_q.shape[1:], x_train_a.shape[1:],y_train=quatrain_model.call(input_tensor_1=x_train_q.shape[1:], input_tensor_2=x_train_a.shape[1:], target_tensor=y_train)
c=x_train_a.shape[1:]
print(c)

# quatrain_model.compile(optimizer='Adam',loss='binary_crossentropy')
# quatrain_model.build(input_shape=[x_train_q.shape[1:], x_train_a.shape[1:]] )
quatrain_model = Transformer(1,1,1,1,1,1,0.01)
quatrain_model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1) , loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False))
# quatrain_model.build(input_shape=[x_train_q.shape[1:], x_train_a.shape[1:],x])
# quatrain_model.summary()
# quatrain_model = get_qa_attention(x_train_q.shape[1:], x_train_a.shape[1:],x)
callback = [keras.callbacks.EarlyStopping(monitor='val_auc', patience=1, mode="max", verbose=0), ]
quatrain_model.fit([x_train_q, x_train_a,y_train],  callbacks=callback, validation_split=0.2,  epochs=10, )
quatrain_model.save('quatrainmodel.h5')

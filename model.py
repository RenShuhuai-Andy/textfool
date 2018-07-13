import spacy
import keras
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Flatten, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from keras.layers.embeddings import Embedding
from keras.constraints import maxnorm
from keras.regularizers import l2
from keras import backend as K


def get_embeddings(vocab):
    max_rank = max(lex.rank + 1 for lex in vocab)  # 删除了 if lex.has_vector
    vectors = np.ndarray((max_rank + 1, vocab.vectors_length), dtype='float32')
    for lex in vocab:
        if lex.has_vector:
            vectors[lex.rank + 1] = lex.vector
    return vectors


vocab_nlp = spacy.load('en', parser=False, tagger=False, entity=False)
print('Preparing embeddings...')
embeddings = get_embeddings(vocab_nlp.vocab)  # shape of embedding is (57393,0)


# 采用keras模型中的sequential模型
def build_model(max_length=1000,
                nb_filters=64,
                kernel_size=3,
                pool_size=2,
                regularization=0.01,
                weight_constraint=2.,
                dropout_prob=0.4,
                clear_session=True):
    if clear_session:
        K.clear_session()

    model = Sequential()  # 序列模型，一系列网络层按顺序构成的栈
    # model中所有layer构成一个list
    model.add(Embedding(
        embeddings.shape[0],
        embeddings.shape[1],
        input_length=max_length,
        trainable=False,
        weights=[embeddings]))

    model.add(Conv1D(nb_filters, kernel_size, activation='relu'))
    model.add(Conv1D(nb_filters, kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size))

    model.add(Dropout(dropout_prob))

    model.add(Conv1D(nb_filters * 2, kernel_size, activation='relu'))
    model.add(Conv1D(nb_filters * 2, kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size))

    model.add(Dropout(dropout_prob))

    model.add(GlobalAveragePooling1D())

    # Dense代表一个全连接层，输出维度是1
    model.add(Dense(1,
                    kernel_regularizer=l2(regularization),
                    kernel_constraint=maxnorm(weight_constraint),
                    activation='sigmoid'))

    # 编译：在训练模型之前，我们需要通过compile来对学习过程进行配置。
    model.compile(
        loss='binary_crossentropy',  # 二分类问题的损失函数
        optimizer='rmsprop',
        metrics=['accuracy'])  # 指标列表：对分类问题，我们一般将该列表设置为metrics=['accuracy']

    return model

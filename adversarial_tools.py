import keras
import spacy
import numpy as np
import tensorflow as tf

from keras import backend as K

from data_utils import extract_features
from paraphrase import perturb_text, _compile_perturbed_tokens

nlp = spacy.load('en', tagger=False, entity=False)


class ForwardGradWrapper:
    '''
    Utility class that computes the gradient of model probability output with respect to model input.
    获取前向导数，前向导数到底是怎么算的？？？
    '''

    def __init__(self, model):
        '''
        :param model: Keras model.
            This code makes a bunch of assumptions about the model:
            - Model has single input
            - Embedding is the first layer
            - Model output is a scalar (logistic regression)
        '''

        input_tensor = model.input
        embedding_tensor = model.layers[0](input_tensor)  # 获取第0层(embedding层)的输入张量
        output_tensor = embedding_tensor
        for layer in model.layers[1:]:
            output_tensor = layer(output_tensor)
        grad_tensor, = tf.gradients(output_tensor, [embedding_tensor])  # output_tensor对embedding_tensor求导
        grad_sum_tensor = tf.reduce_sum(grad_tensor, reduction_indices=2)  # 对梯度张量grad_tensor中的所有元素进行求和，最后得到标量

        self.model = model
        self.input_tensor = input_tensor
        self.grad_sum_tensor = grad_sum_tensor

    def wordwise_grads(self, feature_vectors):
        sess = K.get_session()
        grad_sum = sess.run(self.grad_sum_tensor, feed_dict={
            self.input_tensor: feature_vectors,
            keras.backend.learning_phase(): 0  # 返回训练模式/测试模式(0/1)的flag，以决定当前模型执行于训练模式下还是测试模式下
        })
        return grad_sum


_stats_probability_shifts = []


def adversarial_paraphrase(doc, grad_guide, target, max_length=1000,
                           use_typos=False, verbose=False):
    '''
    Compute a perturbation, greedily choosing the synonyms by maximizing the forward derivative of the model towards target class.
    获取一个扰动单词(doc)，用贪婪算法，找到一个同义词，使替换之后模型对目标类别的前向导数值最大
    '''

    model = grad_guide.model

    x = extract_features([doc], max_length=max_length)[0]  # x为数字化矩阵的第一行，即doc中的第一句
    # model.predict按batch获得输入数据对应的输出，函数的返回值是预测值的numpy array
    y = model.predict(x.reshape(1, -1), verbose=0).squeeze()  # y为原文本的预测输出值
    if verbose:
        print('Prob before', y)

    def halt_condition_fn(perturbed_text):
        '''
        Halt if model output is changed.
        检查模型输出是否改变(是否fool成功)
        '''
        perturbed_x = extract_features([nlp(perturbed_text)],
                                       max_length=max_length)[0]  # 将对抗文本的第一行进行数字化处理，得到perturbed_X
        y = model.predict_classes([perturbed_x.reshape(1, -1)], verbose=0).squeeze()  # 对perturbed_x进行预测
        if y == target:  # 如果fool成功？
            return True
        else:
            return False

    def heuristic_fn(text, candidate):
        '''
        Return the difference between the forward derivative of the original
        word and the candidate substitute synonym, amplified by synonym
        relevance rank.
        返回原单词和替换单词的正向导数
        Yes, this one is pretty bad in terms of performance.
        '''
        doc = nlp(text)
        x = extract_features([doc], max_length=max_length)[0].reshape(1, -1)
        grads = grad_guide.wordwise_grads(x).squeeze()
        index = candidate.token_position
        derivative = grads[index]

        perturbed_tokens = _compile_perturbed_tokens(doc, [candidate])
        perturbed_doc = nlp(' '.join(perturbed_tokens))
        perturbed_x = extract_features(
            [perturbed_doc], max_length=max_length)[0].reshape(1, -1)
        perturbed_grads = grad_guide.wordwise_grads(perturbed_x).squeeze()
        perturbed_derivative = perturbed_grads[index]
        rank = candidate.similarity_rank + 1
        raw_score = derivative - perturbed_derivative
        raw_score *= -1 * target
        return raw_score / rank

    perturbed_text = perturb_text(doc,
                                  use_typos=use_typos,
                                  heuristic_fn=heuristic_fn,
                                  halt_condition_fn=halt_condition_fn,
                                  verbose=verbose)

    perturbed_x = extract_features([nlp(perturbed_text)],
                                   max_length=max_length).reshape(1, -1)
    perturbed_y = model.predict(perturbed_x, verbose=0).squeeze()
    _stats_probability_shifts.append(perturbed_y - y)
    if verbose:
        print('Prob after:', perturbed_y)

    perturbed_y_class = model.predict_classes(perturbed_x, verbose=0).squeeze()
    return perturbed_text, (y, perturbed_y)

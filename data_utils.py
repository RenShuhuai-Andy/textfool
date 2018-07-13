import spacy
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import LabelEncoder

# raw_data_path = './data/twitter_gender_data-01.csv'
raw_data_path = './data/twitter_gender_data.csv'
nlp = spacy.load('en')


def extract_features(docs, max_length):
    '''
    特征处理，返回一个(len(docs),max_length)的矩阵，矩阵中的每行代表docs中的一项文本(即list中的一个元素、一个doc)
    一段/句文本中的单词用对应的token表示，标点符号和空格用0表示。一行不足max_length的用0补足
    存疑：为啥token.rank+1都是1？？？有3个词就是[1,1,1,0,0,...]？？
    :param docs: 形如[text1 | des1, text2 | des2, text4 | des4, text5 | des5, text6 | des6]
    '''
    docs = list(docs)
    X = np.zeros((len(docs), max_length), dtype='int32')
    for i, doc in enumerate(docs):  # i代表下标，doc代表docs中的每一项文本，形如text1 | des1
        j = 0
        for token in doc:  # spacy中的token代表单个标记，即单词、标点符号，不包括空格? 比如text1、|、des1
            # print("token:"+str(token)+"--is_punc:"+str(token.is_punct)+"--is_space:"+str(token.is_space))
            if token.has_vector and not token.is_punct and not token.is_space:  # 如果token有对应的向量且不是标点/空格
                X[i, j] = token.rank + 1  # rank是token词汇类型的顺序ID，用于索引表格，例如用于单词向量。
                j += 1
                if j >= max_length:
                    break
    return X


# 载入推特性别数据
def load_twitter_gender_data(from_cache=False):
    cached_data_path = raw_data_path + '.cached.pkl'

    if from_cache:
        print('Loading data from cache...')
        with open(cached_data_path, 'rb') as f:
            return pickle.load(f)

    max_length = 1000

    print('Loading and preparing data...')
    raw_data = pd.read_csv(raw_data_path, encoding='latin1')
    # print("info:")
    # print(raw_data.info())

    raw_data['text'] = raw_data['text'].apply(str)  # 推特的具体文字内容
    raw_data['description'] = raw_data['description'].apply(str)  # 大概指个人描述，个性签名之类

    # Leave only those rows with 100% confidence, and throw away 'brand' and 'unknown' labels
    # 除去brand/unknown以外，留下100%能确定male/female性别的推特
    raw_data = raw_data[raw_data['gender:confidence'] == 1]
    raw_data = raw_data[raw_data['gender'].apply(
        lambda val: val in ['male', 'female'])]
    print('Raw data with 100% confidence:',
          raw_data.shape)  # raw_data.shape=(10020,26)，原来有19643条。26应该是推特的属性个数，比如性别、创建时间等
    # 把text和description整合起来，数据表中加上一列combined_text，每一项都为text|description
    raw_data['combined_text'] = raw_data.apply(
        lambda row: ' | '.join([row['text'], row['description']]), axis=1)

    # Parse tweet texts
    # 分析推特文本，最后得到的docs是一个list，其中每一个元素都是combined_text中的一项，即某个用户的text|description文本
    docs = list(nlp.pipe(raw_data['combined_text'], batch_size=5000, n_threads=2))

    # Encode labels
    # 将label(male/female)用0/1表示
    label_encoder = LabelEncoder()
    label_encoder.fit(raw_data['gender'])
    y = label_encoder.transform(raw_data['gender'])  # len(y)=10020，y是一个由0、1构成的list，代表性别

    # Pull the raw_data into vectors
    X = extract_features(docs, max_length=max_length)  # 返回一个(10020,1000)的矩阵，把docs中的每一条combined_text都转化为1000维向量

    # Split into train and test sets
    # 获得索引，将数据集随机分为训练集和测试集，测试集占20%
    rs = ShuffleSplit(n_splits=2, random_state=42, test_size=0.2)
    train_indices, test_indices = next(rs.split(X))

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    docs = np.array(docs, dtype=object)  # shape is (10020,)
    docs_train = docs[train_indices]
    docs_test = docs[test_indices]

    numeric_data = X_train, y_train, X_test, y_test
    raw_data = docs_train, docs_test, label_encoder

    with open(cached_data_path, 'wb') as f:
        pickle.dump((numeric_data, raw_data), f)

    # 返回数字化后的数据和原始数据
    return numeric_data, raw_data


# 定义一个load方法的map，可以针对不同数据表写不同的load方法，然后装在下面这个方法中
def load(data_name, *args, **kwargs):
    load_fn_map = {
        'twitter_gender_data': load_twitter_gender_data
    }
    return load_fn_map[data_name](*args, **kwargs)

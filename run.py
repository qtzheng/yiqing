from bert4keras.backend import keras, set_gelu
from bert4keras.bert import build_bert_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.tokenizer import Tokenizer
from keras.layers import *
import os
import re
import pandas as pd
import numpy as np
from itertools import combinations, product
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", 'data', "Data directory where raw data located.")
flags.DEFINE_integer("batch_size", 64, "batch_size")
flags.DEFINE_integer("seq_length", 64, "seq_length")
flags.DEFINE_integer("epochs", 10, "epochs")


class InputFeatures(object):
    def __init__(self, input_ids, token_type_ids, attention_mask, label):
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.label = int(label)


class InputExample(object):
    def __init__(self, category, query1, query2, label):
        self.re_punctuation = '[{}]+'.format(''';'",.!?；‘’“”，。！？''')
        self.category = category
        self.query1 = re.sub(self.re_punctuation, '', query1)
        self.query2 = re.sub(self.re_punctuation, '', query2)
        self.label = int(label)

    def convert_to_features(self, tokenizer, trans=False):
        encode_data = None
        if trans:
            encode_data = tokenizer.encode_plus(self.query2, self.query1, max_length=64, pad_to_max_length=True)
        else:
            encode_data = tokenizer.encode_plus(self.query1, self.query2, max_length=64, pad_to_max_length=True)
        return InputFeatures(encode_data['input_ids'], encode_data['token_type_ids'], encode_data['attention_mask'],
                             self.label)

    def getKey(self):
        return self.category + '@@' + self.query1


class Question(object):
    def __init__(self, category, question):
        self.category = category
        self.question = question
        self.equalQuestions = [question]
        self.notEqualQuestions = []

    def add(self, example):
        if example.label == 1:
            self.equalQuestions.append(example.query2)
        else:
            self.notEqualQuestions.append(example.query2)

    def toExamples(self):
        _examples = []
        par = combinations(self.equalQuestions, 2)
        for text1, text2 in par:
            _examples.append(InputExample(self.category, text1, text2, 1))
        par = product(self.equalQuestions, self.notEqualQuestions)
        for text1, text2 in par:
            _examples.append(InputExample(self.category, text1, text2, 0))
        return _examples


class DataProcess(object):
    def __init__(self, data_path, tokenizer=None):
        self.data_path = data_path
        self.tokenizer = tokenizer

    def getTrainDataSet(self, file_name=None):
        if file_name is None:
            file_name = 'train.csv'
        examples = self._get_examples(os.path.join(self.data_path, file_name))
        features = self._get_features(examples, is_exchange=False)
        return self._get_dataset(features), len(features)

    def getValidDataSet(self, file_name=None):
        if file_name is None:
            file_name = 'dev.csv'
        examples = self._get_examples(os.path.join(self.data_path, file_name))
        features = self._get_features(examples, is_exchange=False)
        return self._get_dataset(features), len(features)

    def getTestDataSet(self, file_name=None):
        if file_name is None:
            file_name = 'test.csv'
        examples = self._get_examples(os.path.join(self.data_path, file_name))
        features = self._get_features(examples, is_exchange=False)
        return self._get_dataset(features), len(features)

    def savePredictData(self, file_name=None):
        if file_name is None:
            file_name = 'result.csv'

    def _get_examples(self, file_name):
        if os.path.exists(file_name):
            data = pd.read_csv(file_name).dropna()
            examples = []
            for i, line in data.iterrows():
                examples.append(InputExample(line['category'], line['query1'], line['query2'], line['label']))
            return examples
        else:
            raise FileNotFoundError('{0} not found.'.format(file_name))

    def _get_features(self, examples, is_exchange=True):
        features = []
        for e in examples:
            features.append(e.convert_to_features(self.tokenizer, False))
            if is_exchange:
                features.append(e.convert_to_features(self.tokenizer, True))
        return features

    def _get_dataset(self, features):
        def gen():
            for ex in features:
                yield (
                    {'input_ids': ex.input_ids, 'attention_mask': ex.attention_mask,
                     'token_type_ids': ex.token_type_ids},
                    ex.label)

        return tf.data.Dataset.from_generator(gen,
                                              ({'input_ids': tf.int32,
                                                'attention_mask': tf.int32,
                                                'token_type_ids': tf.int32},
                                               tf.int64),
                                              ({'input_ids': tf.TensorShape([None]),
                                                'attention_mask': tf.TensorShape([None]),
                                                'token_type_ids': tf.TensorShape([None])},
                                               tf.TensorShape([])))


os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras

import tensorflow as tf

set_gelu('tanh')  # 切换gelu版本

maxlen = FLAGS.seq_length
batch_size = FLAGS.batch_size
config_path = 'model/roberta/bert_config.json'
checkpoint_path = 'model/roberta/bert_model.ckpt'
dict_path = 'model/roberta/vocab.txt'


def load_data(filename):
    D = pd.read_csv(filename).values.tolist()
    return D


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """

    def __init__(self, data, batch_size=32, isTrian=False):
        super().__init__(data, batch_size)
        self.re_punctuation = '[{}]+'.format(''';'\",.!?；‘’“”，。！？''')
        self.isTrian = isTrian

    def __iter__(self, random=False):
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for i in idxs:
            e = self.data[i]
            #             print(text1, text2, label)
            # text1 = re.sub(self.re_punctuation, '', text1)
            # text2 = re.sub(self.re_punctuation, '', text2)

            token_ids, segment_ids = tokenizer.encode(e.query1, e.query2, max_length=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([e.label])
            if self.isTrian:
                token_ids, segment_ids = tokenizer.encode(e.query2, e.query1, max_length=maxlen)
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append([e.label])

            if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 加载预训练模型
bert = build_bert_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    with_pool=True,
    return_keras_model=False,
)

output = Dropout(rate=0.1)(bert.model.output)
output = Dense(units=2,
               activation='softmax',
               kernel_initializer=bert.initializer)(output)
bert.model.trainable = False
model = keras.models.Model(bert.model.input, output)
model.summary()

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(5e-5),  # 用足够小的学习率
    metrics=['accuracy'],
)


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        test_acc = evaluate(valid_generator)
        print(u'test_acc: %.5f\n'
              % test_acc)
        if self.best_val_acc < test_acc:
            result = []
            for x_true, y_true in tests_generator:
                y_pred = model.predict(x_true).argmax(axis=1)
                result = result + y_pred.tolist()
            result = [[i, e] for i, e in enumerate(result)]
            result = np.array(result)
            print('save result')
            result = pd.DataFrame(result, columns=['id', 'label'])
            result.to_csv('result.csv', index=False)
            self.best_val_acc = test_acc


def getTestData(data, batch_size):
    idxs = list(range(len(data)))
    batch_token_ids, batch_segment_ids, batch_labels = [], [], []
    for i in idxs:
        e = data[i]
        token_ids, segment_ids = tokenizer.encode(e.query1, e.query2, max_length=maxlen)
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)
        batch_labels.append([e.label])

        if len(batch_token_ids) == batch_size or i == idxs[-1]:
            batch_token_ids = sequence_padding(batch_token_ids)
            batch_segment_ids = sequence_padding(batch_segment_ids)
            batch_labels = sequence_padding(batch_labels)
            yield [batch_token_ids, batch_segment_ids], batch_labels
            batch_token_ids, batch_segment_ids, batch_labels = [], [], []


if __name__ == '__main__':
    print(tf.__version__)
    dataProcess = DataProcess(FLAGS.data_dir)
    valid_data = dataProcess._get_examples(os.path.join(FLAGS.data_dir, 'dev.csv'))
    train_data = dataProcess._get_examples(os.path.join(FLAGS.data_dir, 'train.csv'))
    tests_data = dataProcess._get_examples(os.path.join(FLAGS.data_dir, 'test.csv'))

    questions = {}
    for e in train_data:
        question = questions.get(e.getKey(), Question(e.category, e.query1))
        question.add(e)
        questions[e.getKey()] = question

    train_data = []
    for value in questions.values():
        train_data = train_data + value.toExamples()

    # 转换数据集
    train_generator = data_generator(train_data, batch_size, isTrian=True)
    valid_generator = data_generator(valid_data, batch_size)
    tests_generator = getTestData(tests_data, batch_size)

    evaluator = Evaluator()
    model.fit_generator(train_generator.forfit(),
                        steps_per_epoch=len(train_generator),
                        epochs=FLAGS.epochs,
                        callbacks=[evaluator], verbose=2)

    print(u'final test acc: %05f\n' % (evaluate(valid_generator)))

import re
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from math import factorial
from itertools import combinations, product


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


if __name__ == '__main__':
    dataProcess = DataProcess('data')
    examples = dataProcess._get_examples('data/dev.csv')
    examples = examples + dataProcess._get_examples('data/train.csv')
    questions = {}
    for e in examples:
        question = questions.get(e.getKey(), Question(e.category, e.query1))
        question.add(e)
        questions[e.getKey()] = question

    print(len(questions))

    examples = []
    for value in questions.values():
        examples = examples + value.toExamples()
        print(len(examples), value.question)
    print(len(examples))

from bert4keras.backend import keras, set_gelu
from bert4keras.bert import build_bert_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.tokenizer import Tokenizer
import pandas as pd
import numpy as np
import os, re
from keras.layers import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras

import tensorflow as tf

set_gelu('tanh')  # 切换gelu版本

maxlen = 64
batch_size = 32
config_path = 'model/roberta/bert_config.json'
checkpoint_path = 'model/roberta/bert_model.ckpt'
dict_path = 'model/roberta/vocab.txt'


def load_data(filename):
    D = pd.read_csv(filename).values.tolist()
    return D


# 加载数据集
all_data = load_data('data/train.csv')
all_data = all_data + load_data('data/dev.csv')
random_order = range(len(all_data))
np.random.shuffle(list(random_order))
train_data = [all_data[j] for i, j in enumerate(random_order) if i % 6 != 1 and i % 6 != 2]
valid_data = [all_data[j] for i, j in enumerate(random_order) if i % 6 == 1]
test_data = [all_data[j] for i, j in enumerate(random_order) if i % 6 == 2]
# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """

    def __init__(self, data, batch_size=32):
        super().__init__(data, batch_size)
        self.re_punctuation = '[{}]+'.format(''';'\",.!?；‘’“”，。！？''')

    def __iter__(self, random=False):
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for i in idxs:
            _, _, text1, text2, label = self.data[i]
            #             print(text1, text2, label)
            text1 = re.sub(self.re_punctuation, '', text1)
            text2 = re.sub(self.re_punctuation, '', text2)

            token_ids, segment_ids = tokenizer.encode(text1, text2, max_length=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])

            token_ids, segment_ids = tokenizer.encode(text2, text1, max_length=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])

            if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
with strategy.scope():
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
        optimizer=Adam(2e-5),  # 用足够小的学习率
        # optimizer=PiecewiseLinearLearningRate(Adam(5e-5), {10000: 1, 30000: 0.1}),
        metrics=['accuracy'],
    )

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)


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
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('best_model.weights')
        test_acc = evaluate(test_generator)
        print(u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n'
              % (val_acc, self.best_val_acc, test_acc))


evaluator = Evaluator()
model.fit_generator(train_generator.forfit(),
                    steps_per_epoch=len(train_generator),
                    epochs=20,
                    callbacks=[evaluator], verbose=2)

model.load_weights('best_model.weights')
print(u'final test acc: %05f\n' % (evaluate(test_generator)))

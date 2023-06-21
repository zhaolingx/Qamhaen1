# -*- coding: utf-8 -*-
# @Time : 2021/3/30 22:56 
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : transformer.py
# @Software: PyCharm
from abc import ABC
import tensorflow as tf
import numpy as np
from experiment.config1 import classifier_config
from experiment.mask import create_padding_mask, create_look_ahead_mask


class PositionalEncoding(tf.keras.layers.Layer):
    """
    Args:
        embedding_dim: 模型的维度，论文默认是512
        seq_length: 文本序列的最大长度
    """

    def __init__(self, embedding_dim, seq_length):
        super(PositionalEncoding, self).__init__()
        # 根据论文给的公式，构造出PE矩阵
        self.pe = np.array([[pos / np.power(10000, 2 * (i // 2) / embedding_dim)
                             for i in range(embedding_dim)] for pos in range(seq_length)])
        # 偶数列使用sin，奇数列使用cos
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])

    @tf.function
    def call(self, inputs):
        # 在这里将词的embedding和位置embedding相加
        position_embed = inputs + self.pe
        return position_embed


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    多头注意力机制
    """
    def __init__(self, embedding_dim):
        super(MultiHeadAttention, self).__init__()
        self.head_num = classifier_config['head_num']

        if embedding_dim % self.head_num != 0:
            raise ValueError(
                'embedding_dim({}) % head_num({}) is not zero. embedding_dim must be multiple of head_num.'.format(
                    embedding_dim, self.head_num))

        self.head_dim = embedding_dim // self.head_num
        self.embedding_dim = embedding_dim
        self.W_Q = tf.keras.layers.Dense(embedding_dim)
        self.W_K = tf.keras.layers.Dense(embedding_dim)
        self.W_V = tf.keras.layers.Dense(embedding_dim)
        self.W_O = tf.keras.layers.Dense(embedding_dim)

    def scaled_dot_product_attention(self, query, key, value, mask):
        """
        缩放点积注意力
        """
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        dk = self.head_dim ** -0.5
        scaled_attention = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention, axis=-1)
        output = tf.matmul(attention_weights, value)
        return output

    def split_head(self, tensor, batch_size):
        tensor = tf.reshape(tensor, (batch_size, -1, self.head_num, self.head_dim))
        tensor = tf.transpose(tensor, perm=[0, 2, 1, 3])
        return tensor

    @tf.function
    def call(self, inputs, mask=None):
        batch_size = tf.shape(inputs)[0]
        query = self.W_Q(inputs)
        key = self.W_K(inputs)
        value = self.W_V(inputs)

        query = self.split_head(query, batch_size)
        key = self.split_head(key, batch_size)
        value = self.split_head(value, batch_size)

        z = self.scaled_dot_product_attention(query, key, value, mask)
        z = tf.reshape(z, (batch_size, -1, self.embedding_dim))
        z = self.W_O(z)
        return z


class PositionWiseFeedForwardLayer(tf.keras.layers.Layer):
    """
    FeedForward层
    """
    def __init__(self, embedding_dim):
        super(PositionWiseFeedForwardLayer, self).__init__()
        hidden_dim = classifier_config['hidden_dim']
        self.dense_1 = tf.keras.layers.Dense(hidden_dim, activation='relu', kernel_initializer='he_uniform')
        self.dense_2 = tf.keras.layers.Dense(embedding_dim)

    @tf.function
    def call(self, inputs):
        output = self.dense_1(inputs)
        output = self.dense_2(output)
        return output


class Encoder(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, dropout_rate):
        super(Encoder, self).__init__()
        self.attention = MultiHeadAttention(embedding_dim)
        self.feed_forward = PositionWiseFeedForwardLayer(embedding_dim)
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout_1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout_2 = tf.keras.layers.Dropout(dropout_rate)

    @tf.function
    def call(self, inputs, training, mask):
        attention_outputs = self.attention(inputs, mask)
        outputs_1 = self.dropout_1(attention_outputs, training=training)
        outputs_1 = self.layer_norm_1(inputs + outputs_1)
        ffn_output = self.feed_forward(outputs_1)
        outputs_2 = self.dropout_2(ffn_output, training=training)
        outputs_2 = self.layer_norm_2(outputs_1 + outputs_2)
        return outputs_2

class Decoder(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, dropout_rate):
        super(Decoder, self).__init__()
        self.masked_attention = MultiHeadAttention(embedding_dim)
        self.attention_1 = MultiHeadAttention(embedding_dim)
        self.attention_2 = MultiHeadAttention(embedding_dim)
        self.feed_forward = PositionWiseFeedForwardLayer(embedding_dim)
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_4 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout_1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout_2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout_3 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout_4 = tf.keras.layers.Dropout(dropout_rate)

    @tf.function
    def call(self, dec_inputs, first_outputs, second_outputs, training, look_ahead_mask, padding_mask):
        masked_attention_outputs = self.masked_attention(dec_inputs, look_ahead_mask)
        outputs_1 = self.dropout_1(masked_attention_outputs, training=training)
        outputs_1 = self.layer_norm_1(dec_inputs + outputs_1)
        attention_outputs_1 = self.attention_1(outputs_1, first_outputs, first_outputs, padding_mask)
        outputs_2 = self.dropout_2(attention_outputs_1, training=training)
        outputs_2 = self.layer_norm_1(attention_outputs_1 + outputs_2)
        attention_outputs_2 = self.attention_2(outputs_2, second_outputs, second_outputs, padding_mask)
        outputs_3 = self.dropout_3(attention_outputs_2, training=training)
        outputs_3 = self.layer_norm_2(attention_outputs_2 + outputs_3)
        ffn_output = self.feed_forward(outputs_3)
        outputs_4 = self.dropout_2(ffn_output, training=training)
        outputs_4 = self.layer_norm_2(ffn_output + outputs_4)
        return outputs_4


class Transformer(tf.keras.Model, ABC):
    """
    Transformer模型
    """

    def __init__(self, vocab_size, embedding_dim, seq_length, num_classes):
        super(Transformer, self).__init__()
        dropout_rate = classifier_config['dropout_rate']
        encoder_num = classifier_config['encoder_num']
        self.embedding = tf.keras.layers.Embedding(vocab_size + 1, embedding_dim, mask_zero=True)
        self.positional_encoder = PositionalEncoding(embedding_dim, seq_length)
        self.encoders = [Encoder(embedding_dim, dropout_rate) for _ in range(encoder_num)]
        self.dense = tf.keras.layers.Dense(num_classes, activation='softmax')
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.avg_pool = tf.keras.layers.GlobalAvgPool1D()

    @tf.function
    def call(self, inputs, training=None):
        mask = create_padding_mask(inputs)
        # embed_inputs = self.embedding(inputs)
        # output = self.positional_encoder(embed_inputs)
        output = self.dropout(inputs, training=training)
        for encoder in self.encoders:
            output = encoder(output, training, mask)
        # output = self.avg_pool(output)
        # output = self.dense(output)
        return output


class Transformer1(tf.keras.Model, ABC):
    def __init__(self, vocab_size, embedding_dim, seq_length, num_classes):
        super(Transformer1, self).__init__()
        dropout_rate = classifier_config['dropout_rate']
        encoder_num = classifier_config['encoder_num']
        decoder_num = classifier_config['decoder_num']

        self.embedding = tf.keras.layers.Embedding(vocab_size + 1, embedding_dim, mask_zero=True)
        self.positional_encoder = PositionalEncoding(embedding_dim, seq_length)
        # self.encoders_1 = [FirstEncoder(embedding_dim, dropout_rate) for _ in range(encoder_num)]
        # self.encoders_2 = [SecondEncoder(embedding_dim, dropout_rate) for _ in range(encoder_num)]
        self.decoders = [Decoder(embedding_dim, dropout_rate) for _ in range(decoder_num - 1)]
        # self.decoders = [Decoder(embedding_dim, dropout_rate) for _ in range(decoder_num)]
        self.dense = tf.keras.layers.Dense(num_classes, activation='softmax')
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.avg_pool = tf.keras.layers.GlobalAvgPool1D()

    @tf.function
    def call(self, inputs, output_1, output_2, training=None, ):
        # mask_2 = create_padding_mask(second_inputs)
        mask_3 = create_look_ahead_mask(tf.shape(inputs)[1])
        mask_4 = create_padding_mask(inputs)
        mask_5 = create_padding_mask(inputs)
        mask_6 = create_padding_mask(inputs)
        # embed_inputs_3 = self.embedding(inputs)

        output_3 = self.positional_encoder(inputs)
        output_3 = self.dropout(output_3, training=training)
        for decoder in self.decoders:
            output_3 = decoder(output_3, output_1, output_2,
                               training,
                               mask_3, mask_5, mask_6)
        # output_3=decoder[self.decoders](output_3, output_1, output_2,
        #                            training,
        #                            mask_3, mask_5, mask_6)

        output = self.avg_pool(output_3)
        output = self.dense(output)
        return output
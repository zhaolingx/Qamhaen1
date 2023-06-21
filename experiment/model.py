# -*- coding: utf-8 -*-

import numpy as np

import tensorflow as tf  # TF2
from experiment.mask import Mask


class Embedder(tf.keras.layers.Layer):
    def __init__(self, d_model, vocab):
        super(Embedder, self).__init__()
        self.emb = tf.keras.layers.Embedding(vocab, d_model)
        self.d_model = d_model
        self.vocab = vocab

    def call(self, x):

        max_len = x.get_shape()[1]
        print('max_len:', max_len)

        # shape == (batch_size, max_len, d_model)
        x = self.emb(x) * tf.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.get_positional_encoding(max_len)
        return x 

    def get_positional_encoding(self, max_len):
        """PE_(pos, 2i) = sin(pos/10000^(2i/d_model))
        PE_(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        """

        pos = np.expand_dims(np.arange(0, max_len), axis=1)
        div_term = np.array([[1 / np.power(10000, (2 * (i//2) / self.d_model)) for i in range(self.d_model)]])       

        pos = pos * div_term

        pe = np.zeros((max_len, self.d_model))
        pe[:, 0:self.d_model//2] = np.sin(pos[:, 0::2])
        pe[:, self.d_model//2:] = np.cos(pos[:, 0::2])

        pe = np.expand_dims(pe, 0)

        print(pe.shape)

        return tf.cast(pe, dtype=tf.float32) 


class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, axis=-1, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.axis = axis

    def build(self, input_shape):
        dim = input_shape[-1]
        self.a_2 = self.add_weight(
            name='a_2',
            shape=(dim,),
            initializer='ones',
            trainable=True)
        self.b_2 = self.add_weight(
            name='b_2',
            shape=(dim,),
            initializer='zeros',
            trainable=True)
        return super(LayerNormalization, self).build(input_shape)
    
    def call(self, inputs, **kwargs):
        mean = tf.reduce_mean(inputs, axis=self.axis, keepdims=True)
        variance = tf.math.reduce_std(inputs, axis=self.axis, keepdims=True)
        epsilon = tf.constant(1e-5)
        normalized_inputs = (inputs - mean) / tf.sqrt(variance + epsilon)
        result = self.a_2 * normalized_inputs + self.b_2
        return result


class ScaledDotProductAttention(tf.keras.layers.Layer):
    """Attention(Q,K,V) = softmax(Q * K.T / sqrt(d_k))*V
    """
    def __init__(self, d_k, dropout):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, query, key, value, mask=None):
        scores = tf.matmul(query, key, transpose_b=True) \
                / tf.sqrt(tf.cast(self.d_k, dtype=tf.float32))
        print('scores:', scores)
        if mask is not None:
            print('mask print', mask)
            scores += (mask * -1e+9) 
        p_attn = tf.nn.softmax(scores, axis=-1)
        return tf.matmul(p_attn, value), p_attn


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, heads, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.h = heads
        self.W_q = tf.keras.layers.Dense(d_model)
        self.W_k = tf.keras.layers.Dense(d_model)
        self.W_v = tf.keras.layers.Dense(d_model)
        self.W_o = tf.keras.layers.Dense(d_model)

        self.scaled_dot_product = ScaledDotProductAttention(self.d_k, dropout) 

    def call(self, query, key, value, mask=None):
        batch_size = tf.shape(query)[0]

        # shape == (batch_size, max_len, d_model)
        query = self.W_q(query)
        key = self.W_k(key)
        value = self.W_v(value)

        # shape == (batch_size, heads, seq_q, d_k)
        query = tf.transpose(tf.reshape(query, (batch_size, -1, self.h, self.d_k)), [0, 2, 1, 3])
        key = tf.transpose(tf.reshape(key, (batch_size, -1, self.h, self.d_k)), [0, 2, 1, 3])
        value = tf.transpose(tf.reshape(value, (batch_size, -1, self.h, self.d_k)), [0, 2, 1, 3])

        x, attn = self.scaled_dot_product(query, key, value, mask=mask)

        x = tf.reshape(tf.transpose(x, [0, 2, 1, 3]), (batch_size, -1, self.h * self.d_k))

        return self.W_o(x), attn


class PositionwiseFeedForward(tf.keras.layers.Layer):
    """FFN(x) = max(0, xW_1+b_1)W_2 + b_2
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.W_1 = tf.keras.layers.Dense(d_ff)
        self.W_2 = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x):
        return self.W_2(self.dropout(tf.nn.relu(self.W_1(x))))


class FirstEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, heads, d_model, d_ff, dropout):
        super(FirstEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(heads, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff) 
        self.layer_norm_1 = LayerNormalization()
        self.layer_norm_2 = LayerNormalization()
        self.dropout_1 = tf.keras.layers.Dropout(dropout)
        self.dropout_2 = tf.keras.layers.Dropout(dropout)

    def call(self, enc_input_1, mask=None):
        x, attn = self.self_attn(enc_input_1, enc_input_1, enc_input_1, mask=mask)
        x = self.dropout_1(x)
        output = self.layer_norm_1(tf.add(enc_input_1, x))

        x = self.feed_forward(output)
        x = self.dropout_2(x)
        # shape == (batch_size, max_len, d_model)
        output = self.layer_norm_2(tf.add(output, x))

        return output


class SecondEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, heads, d_model, d_ff, dropout):
        super(SecondEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(heads, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.layer_norm_1 = LayerNormalization()
        self.layer_norm_2 = LayerNormalization()
        self.dropout_1 = tf.keras.layers.Dropout(dropout)
        self.dropout_2 = tf.keras.layers.Dropout(dropout)

    def call(self, enc_input_2, mask=None):
        x, attn = self.self_attn(enc_input_2, enc_input_2, enc_input_2, mask=mask)
        x = self.dropout_1(x)
        output = self.layer_norm_1(tf.add(enc_input_2, x))

        x = self.feed_forward(output)
        x = self.dropout_2(x)
        # shape == (batch_size, max_len, d_model)
        output = self.layer_norm_2(tf.add(output, x))

        return output


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, heads, d_model, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.masked_self_attn = MultiHeadAttention(heads, d_model, dropout)
        self.self_attn = MultiHeadAttention(heads, d_model, dropout)

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)

        self.layer_norm_1 = LayerNormalization()
        self.layer_norm_2 = LayerNormalization()
        self.layer_norm_3 = LayerNormalization()
        self.layer_norm_4 = LayerNormalization()

        self.dropout_1 = tf.keras.layers.Dropout(dropout)
        self.dropout_2 = tf.keras.layers.Dropout(dropout)
        self.dropout_3 = tf.keras.layers.Dropout(dropout)
        self.dropout_4 = tf.keras.layers.Dropout(dropout)

    def call(self, dec_input, enc_output_1, enc_output_2, look_ahead_mask, padding_mask):
        x, attn_1 = self.masked_self_attn(dec_input, dec_input, dec_input, look_ahead_mask)
        x = self.dropout_1(x)
        output = self.layer_norm_1(tf.add(dec_input, x))

        x, attn_2 = self.self_attn(output, enc_output_1, enc_output_1, padding_mask)
        x = self.dropout_2(x)
        output = self.layer_norm_2(tf.add(output, x))

        x, attn_3 = self.self_attn(output, enc_output_2, enc_output_2, padding_mask)
        x = self.dropout_3(x)
        output = self.layer_norm_3(tf.add(output, x))

        x = self.feed_forward(output)
        x = self.dropout_4(x)
        output = self.layer_norm_4(tf.add(output, x))

        return output, attn_1, attn_2, attn_3
        
    
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, heads, d_ff,
                  input_vocab_size,target_vocab_size, dropout):
        super(Transformer, self).__init__()
        self.num_layers = num_layers

        self.enc_emb = Embedder(d_model, input_vocab_size)
        self.dec_emb = Embedder(d_model, target_vocab_size)

        self.enc_emb_dropout = tf.keras.layers.Dropout(dropout)
        self.dec_emb_dropout = tf.keras.layers.Dropout(dropout)

        self.enc_layers_1 = [FirstEncoderLayer(heads, d_model, d_ff, dropout)
                           for _ in range(num_layers)]

        self.enc_layers_2 = [SecondEncoderLayer(heads, d_model, d_ff, dropout)
                           for _ in range(num_layers)]
        self.dec_layers = [DecoderLayer(heads, d_model, d_ff, dropout) 
                           for _ in range(num_layers)]

        self.linear = tf.keras.layers.Dense(target_vocab_size)

    def call(self, input_tensor_1, input_tensor_2, target_tensor):

        enc_x_1 = self.enc_emb(input_tensor_1)
        # enc_x = self.enc_emb(enc_x_1)
        enc_x_1 = self.enc_emb_dropout(enc_x_1)
        for i in range(self.num_layers):
            enc_x_1 = self.enc_layers_1[i](input_tensor_1)

        enc_x_2 = self.enc_emb(input_tensor_2)
        enc_x_2 = self.enc_emb_dropout(enc_x_2)
        for i in range(self.num_layers):
            enc_x_2 = self.enc_layers_2[i](input_tensor_2)

        dec_x = self.dec_emb(target_tensor)
        dec_x = self.dec_emb_dropout(dec_x)

        for i in range(self.num_layers - 2):
            dec_x, attn_1, attn_2 = self.dec_layers[i](
                    target_tensor, target_tensor,
                    )

        dec_x, attn_1, attn_2 = self.dec_layers[self.num_layers - 2](
                dec_x, enc_x_1,
                )
        dec_x, attn_1, attn_2 = self.dec_layers[self.num_layers - 1](
            dec_x, enc_x_2,
            )

        return self.linear(dec_x)

# def mai(input_tensor_1, input_tensor_2,target_tensor):
#
#
#     num_layers = 3
#     heads = 12
#     d_model = 240
#     d_ff = 240
#     dropout = 0.01
#     target_vocab_size=23718
#
#     # self.enc_emb = Embedder(d_model, input_vocab_size)
#     # self.dec_emb = Embedder(d_model, target_vocab_size)
#
#     # self.enc_emb_dropout = tf.keras.layers.Dropout(dropout)
#     # self.dec_emb_dropout = tf.keras.layers.Dropout(dropout)
#
#     enc_layers_1 = [FirstEncoderLayer(heads, d_model, d_ff, dropout=0.01)
#                            for _ in range(num_layers)]
#
#     enc_layers_2 = [SecondEncoderLayer(heads, d_model, d_ff, dropout)
#                            for _ in range(num_layers)]
#     dec_layers = [DecoderLayer(heads, d_model, d_ff, dropout)
#                            for _ in range(num_layers)]
#
#     linear = tf.keras.layers.Dense(target_vocab_size)
#     for i in range(num_layers):
#         enc_x_1 = enc_layers_1(input_tensor_1)
#
#     for i in range(num_layers):
#         enc_x_2 = enc_layers_2(input_tensor_2)
#
#     for i in range(num_layers - 2):
#         dec_x, attn_1, attn_2 = dec_layers(
#             target_tensor, target_tensor
#
#             )
#
#     dec_x, attn_1, attn_2 = dec_layers[num_layers - 2](
#         dec_x, enc_x_1
#
#         )
#     dec_x, attn_1, attn_2 = dec_layers[num_layers - 1](
#         dec_x, enc_x_2
#        )
#
#     return linear(dec_x)
# def mai():










# -*- coding: utf-8 -*- 
# @Time : 2021/4/6 22:13
# @Author : Stanley  
# @EMail : gzlishouxian@gmail.com
# @File : mask.py  
# @Software: PyCharm
import tensorflow as tf


# # def create_padding_mask(seq):
# #     """
# #     生成mask，mask值为1
# #     """
# #     seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
# #     # [batch_size, 1, 1, seq_len]
# #     # 执行attention计算时，attention_matrix=[batch_size, num_head, seq_len_q, seq_len_k]
# #     return seq[:, tf.newaxis, tf.newaxis, :]
# # class Mask():
# #     """ref: https://www.tensorflow.org/alpha/tutorials/text/transformer#masking
# #     """
# #
# #     @staticmethod
# def create_padding_mask(seq):
#     seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
#     # add extra dimensions so that we can add the padding
#     # to the attention logits.
#     return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
#
#     # @staticmethod
# def create_look_ahead_mask(size):
#     mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
#     return mask  # (seq_len, seq_len)
#
#     # @staticmethod
#     # def create_masks(inp, tar):
#     #     # Encoder padding mask
#     #     enc_padding_mask = Mask.create_padding_mask(inp)
#     #
#     #     # Used in the 2nd attention block in the decoder.
#     #     # This padding mask is used to mask the encoder outputs.
#     #     last_dec_padding_mask = Mask.create_padding_mask(inp)
#     #
#     #     dec_padding_mask = Mask.create_padding_mask(tar)
#     #
#     #     # Used in the 1st attention block in the decoder.
#     #     # It is used to pad and mask future tokens in the input received by
#     #     # the decoder.
#     #     look_ahead_mask = Mask.create_look_ahead_mask(tf.shape(tar)[1])
#     #     dec_target_padding_mask = Mask.create_padding_mask(tar)
#     #     combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
#     #
#     #     return enc_padding_mask, combined_mask, dec_padding_mask, last_dec_padding_mask
# def create_padding_mask(seq):
#     seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
#
#     # add extra dimensions so that we can add the padding
#     # to the attention logits.
#     return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
# def create_look_ahead_mask(size):
#     mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
#     return mask  # (seq_len, seq_len)
#
# def create_masks(inp_1, inp_2, tar):
#     # Encoder padding mask
#     enc_padding_mask_1 = create_padding_mask(inp_1)
#     enc_padding_mask_2 = create_padding_mask(inp_2)
#
#     # Used in the 2nd attention block in the decoder.
#     # This padding mask is used to mask the encoder outputs.
#     last_dec_padding_mask_1 = create_padding_mask(inp_1)
#     last_dec_padding_mask_2 = create_padding_mask(inp_2)
#
#     dec_padding_mask = create_padding_mask(tar)
#
#     # Used in the 1st attention block in the decoder.
#     # It is used to pad and mask future tokens in the input received by
#     # the decoder.
#     look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
#     dec_target_padding_mask = create_padding_mask(tar)
#     combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
#
#     return enc_padding_mask_1, enc_padding_mask_2, combined_mask, dec_padding_mask, last_dec_padding_mask_1,last_dec_padding_mask_2
class Mask():
    """ref: https://www.tensorflow.org/alpha/tutorials/text/transformer#masking
    """

    @staticmethod
    def create_padding_mask(seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

        # add extra dimensions so that we can add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    @staticmethod
    def create_look_ahead_mask(size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)

    @staticmethod
    def create_masks(inp, tar):
        # Encoder padding mask
        enc_padding_mask = Mask.create_padding_mask(inp)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        last_dec_padding_mask = Mask.create_padding_mask(inp)

        dec_padding_mask = Mask.create_padding_mask(tar)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = Mask.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = Mask.create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, combined_mask, dec_padding_mask, last_dec_padding_mask

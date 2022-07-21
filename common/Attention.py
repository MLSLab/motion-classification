# https://www.kaggle.com/qqgeogor/keras-lstm-attention-glove840b-lb-0-043#L51
# Tuned for TF >=2.0

from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense, Lambda, dot, Activation, concatenate


class Attention(Layer):

    def __init__(self, **kwargs):  # , return_attention_weights=False
        super().__init__(**kwargs)

    def __call__(self, hidden_states):  # , return_attention_weights=False
        """
        Many-to-one attention mechanism for Keras.
        @param hidden_states: 3D tensor with shape (batch_size, time_steps, input_dim).
        @return: 2D tensor with shape (batch_size, 128)
        @author: felixhao28.
        https://github.com/philipperemy/keras-attention-mechanism/blob/master/attention/attention.py
        """
        hidden_size = int(hidden_states.shape[2])
        time_steps = int(hidden_states.shape[1])
        # Inside dense layer
        #              hidden_states            dot               W            =>           score_first_part
        # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
        # W is the trainable weight matrix of attention Luong's multiplicative style score
        score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
        #            score_first_part           dot        last_hidden_state     => attention_weights
        # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
        h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
        score = dot([score_first_part, h_t], [2, 1], name='attention_score')
        attention_weights = Activation('softmax', name='attention_weight')(score)
        # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
        context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector')
        pre_activation = concatenate([context_vector, h_t], name='attention_output')
        # attention_vector = Dense(128, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
        attention_vector = Dense(time_steps, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)

        return_attention_weights = True
        print('time_steps in att = {}, return_attention_weights={}'.format(time_steps, return_attention_weights))
        if return_attention_weights:
            return attention_vector, attention_weights
        else:
            return attention_vector
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model

from tensorflow.keras.layers import LSTM, GRU, SimpleRNN
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Input,Dense, Bidirectional
from tensorflow.keras.layers import GlobalAveragePooling1D, MaxPooling1D, Dropout, Flatten

from Attention import Attention

def make_model(model_type, input_shape, n_classes, model_name='model'):
        
    if model_type==0: # LSTM
        model = Sequential(name = model_name)
        model.add(LSTM(n_timesteps, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(n_timesteps))
        model.add(Dropout(rate=0.2))
        model.add(Dense(64, activation='relu')) # model.add(Dense(100, activation='relu'))
        model.add(Dense(n_classes, activation='softmax'))
        
    elif model_type==1: # Conv1D
        model = Sequential(name = model_name)
        model.add(Conv1D(kernel_size=3, strides=1,filters=32, activation='relu', padding='same', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(kernel_size=3, strides=1,filters=64, activation='relu', padding='same'))
        model.add(MaxPooling1D(pool_size=2))

        model.add(GlobalAveragePooling1D())
        model.add(Dropout(rate=0.2)) #rate = 1 - keep_prob
        model.add(Dense(64, activation='relu'))
        model.add(Dense(n_classes, activation='softmax'))

    
    elif args.model==2: # LSTMwAtt (attention-based LSTM)
        inputs = Input(shape=input_shape)
        x = LSTM(n_timesteps, return_sequences=True, input_shape=input_shape)(inputs)
        x = LSTM(n_timesteps, return_sequences=True)(x)
        x, _ = Attention()(x)
        x = Dropout(rate=0.2)(x)
        x = Dense(64, activation='relu')(x)
        y = Dense(n_classes, activation='softmax')(x)

        model = Model(inputs, y, name = model_name[args.model]+"-based")
        #model2 = Model(inputs, [y, att], name = model_name[args.model]+"-based-att")
        
    return model
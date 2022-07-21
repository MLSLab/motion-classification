import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--model', default=1, type=int,
                    help='model to use')
parser.add_argument('--fn_pickle', default='class_watch_L_class5_len100', type=str,
                    help='file name')

parser.add_argument('--cuda_device_order', default='PCI_BUS_ID', type=str,
                    help='CUDA DEVICE ORDER')
parser.add_argument('--cuda_visible_devices', default='3', type=str,
                    help='CUDA VISIBLE DEVICES')

parser.add_argument('--dir_out', default='./out', type=str,
                    help='out directory')

parser.add_argument('--load_pickle', default=True, type=bool)
parser.add_argument('--train', default=True, type=bool) #otherwise load from file

parser.add_argument('--training_epochs', default=100, type=int,
                    help='training_epochs')

parser.add_argument('--batch_size', default=100, type=int,
                    help='batch_size')



args = parser.parse_args(args=[])

from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Bidirectional
from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K


import glob
import tensorflow as tf
import numpy as np
import pandas as pd
import time
from tqdm import tqdm 
import pickle
import matplotlib.pyplot as plt

from ml_util import get_time_string
import dataset as motion

base_dir_out = args.dir_out

cuda_dev = args.cuda_device_order
cuda_vis = args.cuda_visible_devices

loadFromPickle = args.load_pickle


tstring = get_time_string()

model_name = ('LSTM', 'Conv1D', 'LSTMwAtt')

#config
args.fn_pickle = './pickle/class_watch_L_class5_len100_pub.pickle' 
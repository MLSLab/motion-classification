import os
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


### Files
def make_sure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
        
##### Misc.
def exit():
    os._exit(0)


def get_time_string(m = -1):
    if m==0:
        s1 = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    else:
        s1 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    return s1


#### ML
def convert_to_one_hot(Y, C):
    Y = np.array(int(Y))
    Y = np.eye(C)[Y.reshape(-1)]
    return Y[0]

'''
o1 = convert_to_one_hot(np.array(1),2)
o2 = convert_to_one_hot(np.array(0),2)

print(o1)
print(o2)

'''

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # https://raw.githubusercontent.com/kobiso/CBAM-keras/master/utils.py
    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def plot_confusion_matrix(actual, predicted, classes, title='Confusion Matrix', normalize=False,
                            hide_classname=False, figsize=(4, 4), dpi=36, cmap=plt.cm.viridis, outdir='out/'):
    import itertools
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if not normalize:
        conf_matrix = pd.crosstab(actual, predicted) 
    else:
        conf_matrix = pd.crosstab(actual, predicted).apply(lambda r: r / r.sum(), axis=1)
        
    if np.shape(conf_matrix) != (len(classes), len(classes)):
        print(np.shape(conf_matrix))
        for i in range(len(classes)):
            if i not in conf_matrix.columns:
                conf_matrix[i] = 0.00
        conf_matrix = conf_matrix[[i for i in range(len(classes))]]

    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    ax = plt.gca()
    im = ax.imshow(conf_matrix, aspect=1.0, interpolation='nearest',cmap=cmap)
    
    plt.title(title, size=30)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05, aspect=20)
    plt.colorbar(im, cax=cax)
    
    if hide_classname:
        classes2 = []
        for k, _ in enumerate(classes):
            classes2.append('C{}'.format(k))
        classes = classes2
    
    tick_marks = np.arange(len(classes))

    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=25)
    ax.set_xticklabels(classes, fontsize=25)
    
    fmt = '.2f' if normalize else 'd'
    thresh = conf_matrix.max() / 2.
    thresh = 0.5
    
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        if conf_matrix[j][i] > thresh: #thresh[j]:
            color="white" 
        
        else :
            color = "black"  
        ax.text(j, i, format(conf_matrix[j][i], fmt), ha="center", va="center", color=color, fontsize=25)
        
    ax.grid(False)
    plt.savefig('{}/final_{}.png'.format(outdir, title), bbox_inches='tight')
    plt.savefig('{}/final_{}.svg'.format(outdir, title), bbox_inches='tight')
    
    return fig
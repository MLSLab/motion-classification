import numpy as np
import glob
import matplotlib.pyplot as plt

import random
import time

def convert_to_one_hot(Y, C):
    Y = np.array(int(Y))
    Y = np.eye(C)[Y.reshape(-1)]
    return Y[0]

def swapCols(dat):
    '''
    x = dat[:,0]
    yData = dat[:,9-1] #% SND
    yData2 = dat[:,8-1] # button

    yDataAcc = dat[:, 2-1:4]
    yDataGyr = dat[:, 5-1:7]
    '''
    v = np.array(dat)
    v[:, 0] = v[:, 0] - v[0, 0]
    v[:, 0] = v[:, 0] / 1000000 / 1000  # based on System.nanoTime()

    t = np.expand_dims(v[:, 0], axis=1)
    btn = np.expand_dims(v[:, 8 - 1], axis=1)
    snd = np.expand_dims(v[:, 9 - 1], axis=1)

    datout = np.hstack((t, btn, snd, dat[:, 2 - 1:4], dat[:, 5 - 1:7]))
    return datout


def getSample(rawdata, seq_length, nSample, targetv, idx_btn=7, allowAllZeros=False):
    cntok = 0
    # to_Sample = 10
    t0 = time.time()
    # print(seq_length)
    vdata_collect = []
    while True:
        vidx = random.sample(range(1, rawdata.shape[0] - seq_length - 1), 1)

        d1 = rawdata[vidx[0]:vidx[0] + seq_length]

        if not allowAllZeros and np.sum(d1[:, idx_btn]) == 0:
            continue

        bOk = d1[-1, idx_btn] == targetv

        # print(d1.shape)
        if bOk:
            elapsed = time.time() - t0
            vdata_collect.append(d1)
            # plt.plot(d1[:,7])

            cntok = cntok + 1

            if cntok == nSample:
                # plt.title('{} {}sec'.format(cntok, elapsed))
                # print('{} samples {}sec'.format(cntok, elapsed))
                # plt.ylim([-2, 7])
                break
    return vdata_collect, elapsed


def getSampleInDir(filelists, seq_length, nSample, targetv):
    filelists = np.array(filelists)
    cntok = 0
    # to_Sample = 10
    t0 = time.time()
    # print(seq_length)
    vall = []

    for i in range(len(filelists)):
        for j in range(len(filelists[i])):
            print('{}\t{}\t{}'.format(i, j, filelists[i][j]))
            fn = filelists[i][j]
            arr1 = np.loadtxt(fn, dtype='float')
            if len(vall) == 0:
                vall = arr1

            else:
                vall = np.vstack((vall, arr1))
            print('Loaded {}: {}'.format(fn, arr1.shape))

    return np.array(vall)


def GetListFilesSubdirPattern(destpath, pattern, skip_begins_with="__", verbose=False):
    '''
    listc = glob.glob(fnfmt)
    is compatible with
    listc = GetListFilesSubdir(fnfmt)
    listc = GetListFilesSubdirPattern('../SMNet/c0/*.txt', pattern=["o5", "o-5", "o10"])
    '''
    import os
    dir1 = os.path.dirname(destpath)
    fnfmt1 = os.path.basename(destpath)[0:]  # e.g. *.txt
    # print(dir1, fnfmt1)

    destpath = dir1
    ext = fnfmt1

    filelist = []
    #print(pattern)
    for path, subdirs, full_path in os.walk(destpath):
        # print(path, subdirs)
        subdirname = path.split('/')[-1]

        extractstr = subdirname[:len(skip_begins_with)]
        # print('>>> ' + subdirname)
        # print(full_path)
        # print(subdirname, extractstr,skip_begins_with, extractstr == skip_begins_with )
        if skip_begins_with and extractstr == skip_begins_with:
            print('Skipping....directory {}'.format(subdirname))
            continue

        path1 = os.path.join(path, ext)  # '*_o[0|5].txt'
        # print(path1)
        dir2 = glob.glob(path1)

        if not pattern:
            filelist = filelist + dir2
            continue

        bbb = []

        for diri in dir2:
            file_name = os.path.split(diri)[1]
            cnt_files = 0
            # print(file_name)
            for pat in pattern:
                if pat in file_name:

                    bbb.append(diri)
                    if verbose:
                        print(diri)
                        cnt_files = cnt_files+1
                    break
            if verbose:
                if cnt_files>0:
                    print('---Added {} files'.format(cnt_files))
        filelist = filelist + bbb
        # print(np.shape(bbb), np.shape(filelist))

    return filelist

def GetListFilesSubdir2(destpath, ext="", skip_begins_with="__"):
    '''
    listc = glob.glob(fnfmt)
    is compatible with
    listc = GetListFilesSubdir(fnfmt)

    destpath supports a regex
    '''
    import os

    if not ext:
        dir1 = os.path.dirname(destpath)
        fnfmt1 = os.path.basename(destpath)[0:] # e.g. *.txt
        #print(dir1, fnfmt1)

        destpath = dir1
        ext = fnfmt1

    #print(destpath, ext)
    filelist = []

    for path, subdirs, full_path in os.walk(destpath):
        #print(path, subdirs)
        subdirname = path.split('/')[-1]

        extractstr = subdirname[:len(skip_begins_with)]
        #print('>>> ' + subdirname)
        #print(full_path)
        #print(subdirname, extractstr,skip_begins_with, extractstr == skip_begins_with )
        if skip_begins_with and extractstr == skip_begins_with :
            print('Skipping....directory {}'.format(subdirname))
            continue

        path1 = os.path.join(path, ext) #'*_o[0|5].txt'
        #print(path1)
        bbb = glob.glob(path1)
        filelist = filelist + bbb
        #print(np.shape(bbb), np.shape(filelist))

    return filelist

def GetListFilesSubdir(destpath, ext="", skip_begins_with="__"):
    '''
    listc = glob.glob(fnfmt)
    is compatible with
    listc = GetListFilesSubdir(fnfmt)

    Use GetListFilesSubdir2 instead !
    '''
    import os

    if not ext:
        dir1 = os.path.dirname(destpath)
        fnfmt1 = os.path.basename(destpath)[1:] # e.g. *.txt
        #print(dir1, fnfmt1)

        destpath = dir1
        ext = fnfmt1

    filelist = []

    for path, subdirs, full_path in os.walk(destpath):
        subdirname = path.split('/')[-1]

        extractstr = subdirname[:len(skip_begins_with)]
        #print(subdirname, extractstr,skip_begins_with, extractstr == skip_begins_with )
        if skip_begins_with and extractstr == skip_begins_with :
            print('Skipping....directory {}'.format(subdirname))
            continue
        for filename in full_path:
            f = os.path.join(path, filename)
            if os.path.isfile(f):
                if filename.endswith(ext):
                    filelist.append(f)
    return filelist

def ListFilesInSubDir(filelists):
    filelists = np.array(filelists)
    cntok = 0
    # to_Sample = 10
    t0 = time.time()
    # print(seq_length)
    vdata_collect = []

    for i in range(len(filelists)):
        for j in range(len(filelists[i])):
            print('{}\t{}\t{}'.format(i, j, filelists[i][j]))
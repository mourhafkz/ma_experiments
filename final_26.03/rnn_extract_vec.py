#!/usr/bin/env python3
import glob
import os
import gzip
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import classification_report
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

archimob_dialect_doc = {'1007': ['NW'], '1008': ['LU'], '1044': ['BS'], '1048': ['GL'], '1053': ['NW'],
                        '1055': ['ZH'], '1063': ['AG'], '1073': ['BL'], '1075': ['BS'], '1082': ['ZH'],
                        '1083': ['ZH'], '1087': ['ZH'], '1121': ['BE'], '1138': ['LU'], '1142': ['BE'],
                        '1143': ['ZH'], '1147': ['AG'], '1163': ['AG'], '1170': ['BE'], '1188': ['ZH'], '1189': ['ZH'],
                        '1195': ['LU'], '1198': ['SG'], '1203': ['BE'], '1205': ['SH'], '1207': ['GL'], '1209': ['SZ'],
                        '1212': ['VS'], '1215': ['BE'], '1224': ['BS'], '1225': ['ZH'], '1228': ['ZH'], '1235': ['LU'],
                        '1240': ['GR'], '1244': ['ZH'], '1248': ['AG'], '1255': ['UR'], '1259': ['AG'], '1261': ['LU'],
                        '1263': ['BS'], '1270': ['ZH'], '1295': ['AG'], '1300': ['ZH']}


def read_data(filename, datadir='data', verbose=True,
              max_instances=5000, seq_len=400, truncate=True, pad=True):
    pattern = os.path.join(datadir, filename)
    d, cvid, labels = [], [], []
    for path in glob.glob(pattern):
        with gzip.open(path, 'rt') as f:
            if verbose:
                print(path, flush=True, end="")
                maxlen = 0
                seqlen = []
            mfcc, id_, lbls = [], [], []
            for line in f:
                if line.startswith('common_voice'):
                    if max_instances and len(mfcc) >= max_instances:
                        break
                    currid = line.strip(' [\t\n\r')
                    id_.append(currid)
                    lbls.append(currid.split("_")[2])
                    seq = []
                    mfcc.append(seq)
                elif line.startswith('d'):
                    if max_instances and len(mfcc) >= max_instances:
                        break
                    currid = line.strip(' [\t\n\r')
                    id_.append(currid)
                    id_tofind = currid.split("_")[0][1:5]
                    lbls.append(archimob_dialect_doc[id_tofind][0])
                    seq = []
                    mfcc.append(seq)
                elif line.startswith('1082'):
                    if max_instances and len(mfcc) >= max_instances:
                        break
                    currid = line.strip(' [\t\n\r')
                    id_.append(currid)
                    lbls.append('ZH')
                    seq = []
                    mfcc.append(seq)

                else:
                    if verbose:
                        seqlen.append(len(seq))
                    if ']' in line:
                        if pad and len(seq) < seq_len:
                            seq.extend([[0] * len(seq[0])] * (seq_len - len(seq)))
                    line = line.strip('] \n')
                    if not truncate or len(seq) < seq_len:
                        seq.append([float(x) for x in line.split()])
            d.extend(mfcc)
            cvid.extend(id_)
            # labels.extend([label]*len(mfcc))
            # labels.extend(lbls)
            labels = lbls
            if verbose:
                print(f" {len(mfcc)} instances, "
                      f"max {max(seqlen)} frames, "
                      f"{sum(seqlen) / len(seqlen):.2f} frames average", flush=True)
    return d, labels, cvid

def predict_export(features, labels, model, filename):
  pred_features = loaded_model.predict(features, batch_size=1024)
  wt = open(filename, 'w')
  for index, pred in enumerate(pred_features):
    vector = pred.flatten().tolist()
    vector.append(labels[index])
    wt.write(','.join(map(str,vector))+"\n")





if __name__ == "__main__":
    loaded_model = tf.keras.models.load_model('GRU_BI_2_Layers_best')

    # Check its architecture
    loaded_model.summary()

    # loading all instances 113000
    feat, labels, cvid = read_data(f'commonvoice-mfcc.txt.gz', datadir='./', max_instances=0)

    predict_export(feat, labels, loaded_model, 'commonvoice_mfcc_vectors.csv')

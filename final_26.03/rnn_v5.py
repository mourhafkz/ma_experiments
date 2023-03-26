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
import matplotlib.pyplot as plt



def find_class_weights(y_labels):
  class_weights = class_weight.compute_class_weight(
                                          class_weight = "balanced",
                                          classes = np.unique(y_labels),
                                          y = y_labels
                                      )
  label2id = {k:i for i, k in enumerate(sorted(set(y_labels)))}
  class_weights = dict(zip(list(label2id.values()), class_weights))
  return class_weights


archimob_dialect_doc = {'1007': ['NW'], '1008': ['LU'], '1044': ['BS'], '1048': ['GL'], '1053': ['NW'], 
                        '1055': ['ZH'], '1063': ['AG'], '1073': ['BL'], '1075': ['BS'], '1082': ['ZH'], 
                        '1083': ['ZH'], '1087': ['ZH'], '1121': ['BE'], '1138': ['LU'], '1142': ['BE'], 
                        '1143': ['ZH'], '1147': ['AG'], '1163': ['AG'], '1170': ['BE'], '1188': ['ZH'], '1189': ['ZH'], '1195': ['LU'], '1198': ['SG'], '1203': ['BE'], '1205': ['SH'], '1207': ['GL'], '1209': ['SZ'], '1212': ['VS'], '1215': ['BE'], '1224': ['BS'], '1225': ['ZH'], '1228': ['ZH'], '1235': ['LU'], '1240': ['GR'], '1244': ['ZH'], '1248': ['AG'], '1255': ['UR'], '1259': ['AG'], '1261': ['LU'], '1263': ['BS'], '1270': ['ZH'], '1295': ['AG'], '1300': ['ZH']}
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


def remove_classes(labels, feat):
    # find removable classes based on condition
    removables = [label for label, count in Counter(labels).items() if count <= 10]
    # collect their indices
    removables_indices = [index for index, label in enumerate(labels)
                          for rem in removables
                          if label in rem and label == rem]
    # remove labels and features by indeces
    for index in sorted(removables_indices, reverse=True):
        del labels[index]
        del feat[index]
    return labels, feat


def process_feature_array(trn_feat, dev_feat, trn_labels, dev_labels):
    print(Counter([len(seq) for seq in trn_feat]))
    trn_feat = np.array(trn_feat)
    dev_feat = np.array(dev_feat)
    print(trn_feat.shape, dev_feat.shape)

    label2id = {k: i for i, k in enumerate(sorted(set(trn_labels)))}
    id2label = {k: i for i, k in label2id.items()}
    print(label2id, id2label)

    trn_y = np.array([label2id[x] for x in trn_labels])
    dev_y = np.array([label2id[x] for x in dev_labels])

    # _save_feature_df(trn_feat, dev_feat, trn_y, dev_y)
    return trn_feat, dev_feat, trn_y, dev_y, label2id, id2label



def getBestModel(trn_feat, dev_feat, trn_y, dev_y, label2id, id2label, dev_labels, class_weights, epochs, learning_rate=0.001):
    # preprocess
    # trn_feat, dev_feat, trn_y, dev_y = load_feature_array()
    # Models
    ## input layer
    inp = tf.keras.Input(shape=trn_feat.shape[1:])

    ## GRU_BI_2_Layers
    rnn = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(300, return_sequences=True))(inp)
    rnn = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(300))(rnn)
    dense = tf.keras.layers.Dense(300, activation='tanh')(rnn)
    GRU_BI_2_Layers_dense = tf.keras.layers.Dense(300, activation='tanh', name='feat')(dense)
    out = tf.keras.layers.Dense(len(label2id), activation='softmax')(GRU_BI_2_Layers_dense)
    GRU_BI_2_Layers = tf.keras.Model(inputs=inp, outputs=out)

    ## LSTM_BI_2_Layers
    rnn = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(300, return_sequences=True))(inp)
    rnn = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(300))(rnn)
    dense = tf.keras.layers.Dense(300, activation='tanh')(rnn)
    LSTM_BI_2_Layers_dense = tf.keras.layers.Dense(300, activation='tanh', name='feat')(dense)
    out = tf.keras.layers.Dense(len(label2id), activation='softmax')(LSTM_BI_2_Layers_dense)
    LSTM_BI_2_Layers = tf.keras.Model(inputs=inp, outputs=out)

    # ## GRU_4_Layers
    # rnn = tf.keras.layers.GRU(100, return_sequences=True)(inp)
    # rnn = tf.keras.layers.GRU(100, return_sequences=True)(rnn)
    # rnn = tf.keras.layers.GRU(100, return_sequences=True)(rnn)
    # rnn = tf.keras.layers.GRU(100)(rnn)
    # dense = tf.keras.layers.Dense(100, activation='tanh')(rnn)
    # GRU_4_Layers_dense = tf.keras.layers.Dense(100, activation='tanh')(dense)
    # out = tf.keras.layers.Dense(len(label2id), activation='softmax')(GRU_4_Layers_dense)
    # GRU_4_Layers = tf.keras.Model(inputs=inp, outputs=out)

    # ## LSTM_4_Layers
    # rnn = tf.keras.layers.LSTM(100, return_sequences=True)(inp)
    # rnn = tf.keras.layers.LSTM(100, return_sequences=True)(rnn)
    # rnn = tf.keras.layers.LSTM(100, return_sequences=True)(rnn)
    # rnn = tf.keras.layers.LSTM(100)(rnn)
    # dense = tf.keras.layers.Dense(100, activation='tanh')(rnn)
    # LSTM_4_Layers_dense = tf.keras.layers.Dense(100, activation='tanh')(dense)
    # out = tf.keras.layers.Dense(len(label2id), activation='softmax')(LSTM_4_Layers_dense)
    # LSTM_4_Layers = tf.keras.Model(inputs=inp, outputs=out)

    # ## GRU_BI_4_Layers
    # rnn = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(100, return_sequences=True))(inp)
    # rnn = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(100, return_sequences=True))(rnn)
    # rnn = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(100, return_sequences=True))(rnn)
    # rnn = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(100))(rnn)
    # dense = tf.keras.layers.Dense(100, activation='tanh')(rnn)
    # GRU_BI_4_Layers_dense = tf.keras.layers.Dense(100, activation='tanh')(dense)
    # out = tf.keras.layers.Dense(len(label2id), activation='softmax')(GRU_BI_4_Layers_dense)
    # GRU_BI_4_Layers = tf.keras.Model(inputs=inp, outputs=out)

    # ## LSTM_BI_4_Layers
    # rnn = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100, return_sequences=True))(inp)
    # rnn = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100, return_sequences=True))(rnn)
    # rnn = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100, return_sequences=True))(rnn)
    # rnn = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100))(rnn)
    # dense = tf.keras.layers.Dense(100, activation='tanh')(rnn)
    # LSTM_BI_4_Layers_dense = tf.keras.layers.Dense(100, activation='tanh')(dense)
    # out = tf.keras.layers.Dense(len(label2id), activation='softmax')(LSTM_BI_4_Layers_dense)
    # LSTM_BI_4_Layers = tf.keras.Model(inputs=inp, outputs=out)

    # ## GRU_LSTM_BI_2
    # rnn = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(100, return_sequences=True))(inp)
    # rnn = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100))(rnn)
    # dense = tf.keras.layers.Dense(100, activation='tanh')(rnn)
    # GRU_LSTM_BI_2_dense = tf.keras.layers.Dense(100, activation='tanh')(dense)
    # out = tf.keras.layers.Dense(len(label2id), activation='softmax')(GRU_LSTM_BI_2_dense)
    # GRU_LSTM_BI_2 = tf.keras.Model(inputs=inp, outputs=out)

    # keep track of the final layers
    encoder_layers = {
                    #'GRU_BI_2_Layers_dense': GRU_BI_2_Layers_dense,
                      'LSTM_BI_2_Layers_dense': LSTM_BI_2_Layers_dense,
                      # 'GRU_4_Layers_dense': GRU_4_Layers_dense,
                      #'LSTM_4_Layers_dense': LSTM_4_Layers_dense,
                      #'GRU_BI_4_Layers_dense': GRU_BI_4_Layers_dense,
                      #'LSTM_BI_4_Layers_dense': LSTM_BI_4_Layers_dense,
                      #'GRU_LSTM_BI_2_dense': GRU_LSTM_BI_2_dense
                      }

    # collect models
    models = {
            #'GRU_BI_2_Layers': GRU_BI_2_Layers,
              'LSTM_BI_2_Layers': LSTM_BI_2_Layers,
              #'GRU_4_Layers': GRU_4_Layers,
              #'LSTM_4_Layers': LSTM_4_Layers,
              #'GRU_BI_4_Layers': GRU_BI_4_Layers,
              #'LSTM_BI_4_Layers': LSTM_BI_4_Layers,
              #'GRU_LSTM_BI_2': GRU_LSTM_BI_2
              }

    predictions = {}
    accuracy = {}
    f1 = {}
    for model in models:
        print(model, "_________________________________________________")
        models[model].compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate = learning_rate),
            metrics=["accuracy"]
        )

        history = models[model].fit(trn_feat, trn_y,
                          validation_data=(dev_feat, dev_y),
                          batch_size=512,
                          epochs=epochs,
                          class_weight=class_weights)

        predictions[model] = models[model].predict(dev_feat, batch_size=1024).argmax(axis=1)
        pred_labels = [id2label[x] for x in predictions[model]]

        print(classification_report(dev_labels, pred_labels))
        accuracy[model] = accuracy_score(dev_labels, pred_labels)
        f1[model] = f1_score(dev_labels, pred_labels, average='macro')
        print(models[model].summary())
            
        # plt.plot(history.history['accuracy'])
        # plt.plot(history.history['val_accuracy'])
        # plt.title('model accuracy')
        # plt.ylabel('accuracy')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'val'], loc='upper left')
        # plt.show()
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()
        plt.savefig(model+'_accu.png', bbox_inches='tight')
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()
        plt.savefig(model+'_loss.png', bbox_inches='tight')

            
        temp_model = tf.keras.Model(inputs=models[model].input,
                                 outputs=models[model].get_layer("feat").output)
        
        temp_model.save(model+"_temp")


    print("accuracy scores:", accuracy)
    print("macro average f1:", f1)
    best_model = max(f1, key=f1.get)
    print('Best Model by macro average: ', best_model)

    # extract and save model
    inter_model = tf.keras.Model(inputs=models[best_model].input,
                                 outputs=models[best_model].get_layer("feat").output)
    print(inter_model.summary())

    inter_model.save(best_model+"_best")
    return inter_model

if __name__ == "__main__":
  # loading all instances 113000
  feat, labels, cvid = read_data(f"commonvoice-mfcc.txt.gz", datadir='./', max_instances=0)
  # remove small classes
  labels, feat = remove_classes(labels, feat)

  trn_feat, dev_feat, trn_labels, dev_labels = train_test_split(feat, labels,
                                                                stratify=labels,
                                                                test_size=0.5, random_state=42)

  class_weights = find_class_weights(labels)
  print("class weights:", class_weights)

  trn_feat, dev_feat, trn_y, dev_y, label2id, id2label = process_feature_array(trn_feat, dev_feat, trn_labels, dev_labels)

  inter_model = getBestModel(trn_feat, dev_feat, trn_y, dev_y, label2id, id2label, dev_labels, class_weights, 50, 0.001)




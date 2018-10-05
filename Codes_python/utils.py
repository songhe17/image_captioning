import pickle
import tensorflow as tf
import numpy as np
def load_data(caption_path,anno_path,feats_path,word_to_idx_path):
    print('Loading...')
    features = []
    with open(anno_path, 'rb') as f:
        annotations = pickle.load(f)
    with open(caption_path, 'rb') as f:
        captions = pickle.load(f)
    with open(feats_path,'rb') as f:
        feats = pickle.load(f)
    with open(word_to_idx_path,'rb') as f:
        word_to_idx = pickle.load(f)
    batch_size = 0
    for j in range(len(annotations['file_name'])):
        key = 'COCO_train2014_' + annotations['file_name'][j][39:-4]
        if key in feats.keys():
            batch_size += 1
    print('Overall captions:' + str(batch_size))
    for i in range(batch_size):
        key = 'COCO_train2014_' + annotations['file_name'][i][39:-4]
        features.append(feats[key])
    #features = tf.contrib.layers.batch_norm(features)
    print('Loading finished')
    return captions[:batch_size][:],features,word_to_idx,batch_size

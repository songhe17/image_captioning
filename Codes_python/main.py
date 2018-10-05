from rnn_model import rnn
from vgg16 import encoder
from utils import load_data
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()
def main():
    #trythisone = encoder('/Users/songhewang/Desktop/data_for_IC/train_images/','/Users/songhewang/Desktop/data_for_IC/test_600_train.pkl')
    #trythisone._extract_feats()
    captions,features,word_to_idx,batch_size = load_data('/Users/songhewang/Desktop/data_for_IC/train/train.captions.pkl',
    '/Users/songhewang/Desktop/data_for_IC/train/train.annotations.pkl',
    '/Users/songhewang/Desktop/data_for_IC/test_600_train.pkl',
    '/Users/songhewang/Desktop/data_for_IC/train/word_to_idx.pkl')
    decoder = rnn(word_to_idx,[49,512],512,1024,17,captions,batch_size)
    loss,_ = decoder.train()
    optimizer = decoder.optimizer()
    train = optimizer.minimize(loss)
    test_prediction = decoder.test()
    with tf.Session() as sess:
        print('Finish constructing decoder')
        sess.run(tf.global_variables_initializer())
        sess.run(train,feed_dict = {decoder.annotations:features})














if __name__ == "__main__":
    main()

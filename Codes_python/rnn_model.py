import numpy as np
import tensorflow as tf

class rnn(object):

    def __init__(self,word_to_idx,dim_feats,dim_emb,dim_hidden,caption_len,captions,batch_size):
        self.word_to_idx = word_to_idx
        self.num_feats = dim_feats[0]
        self.dim_feats = dim_feats[1]
        self.dim_emb = dim_emb
        self.dim_hidden = dim_hidden
        self.vocabulary_len = len(word_to_idx)
        self.caption_len = caption_len
        self.batch_size = batch_size
        self.annotations = tf.placeholder(tf.float32,[self.batch_size,self.num_feats,self.dim_feats],'annotations')
        self.captions = captions
        self.test = False
    def _get_initial_input(self):
        with tf.variable_scope('ini_ch'):
            annotations = self.annotations
            annotations_mean = tf.reduce_mean(annotations,2)#[400000,49]
            annotations_mean = tf.nn.dropout(annotations_mean,keep_prob = 0.5)
            c = tf.layers.dense(annotations_mean,self.dim_hidden)#[400000,1024]
            h = tf.layers.dense(annotations_mean,self.dim_hidden)#[400000,1024]
            return c,h

    def _attend(self,h,annotations,reuse):
        '''
        shape of h: (1,512)
        shape of annotations:(49,512)
        shape of alpha:[1,49]
        '''
        with tf.variable_scope('attend',reuse = reuse):
            at_logits1 = tf.layers.dense(annotations,1)#[40000,49,1]
            at_logits1 = tf.reshape(at_logits1,[-1,self.num_feats])#[400000,49]
            at_logits2 = tf.layers.dense(h,self.num_feats)#[400000,49]
            at_alpha = tf.nn.softmax(at_logits1 + at_logits2)
            return at_alpha

    def _word_embedding(self,inputs,reuse):
        with tf.variable_scope('word_embedding',reuse = reuse):
            embedding_matrix = tf.get_variable(name = 'emb_matrx',
            shape = [self.vocabulary_len,self.dim_emb],
            initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0),
            dtype = tf.float32)
            outputs = tf.nn.embedding_lookup(embedding_matrix,inputs,'captions')
            return outputs

    def _decode_lstm(self,x,h,context,reuse):
        with tf.variable_scope('decode',reuse = reuse):
            w_h = tf.get_variable(name = 'wh',shape = [self.dim_hidden,self.dim_emb],initializer = tf.contrib.layers.xavier_initializer(),dtype = tf.float32)
            w_context = tf.get_variable(name = 'wc',shape = [self.dim_feats,self.dim_emb],initializer = tf.contrib.layers.xavier_initializer(),dtype = tf.float32)
            w_L = tf.get_variable(name = 'wl',shape = [self.dim_emb,self.vocabulary_len],initializer = tf.contrib.layers.xavier_initializer(),dtype = tf.float32)
            logits = tf.matmul(tf.matmul(h,w_h)+tf.matmul(context,w_context)+x,w_L)
            logits = tf.nn.softmax(logits)
            return logits

    def train(self):
        #features = tf.contrib.layers.batch_norm()
        norm_annotations = tf.contrib.layers.batch_norm(self.annotations)
        c,h = self._get_initial_input()
        captions = self.captions
        lstm = tf.nn.rnn_cell.BasicLSTMCell(num_units = self.dim_hidden)
        lstm = tf.nn.rnn_cell.DropoutWrapper(lstm,
                input_keep_prob = 0.5,
                output_keep_prob = 0.5,
                state_keep_prob = 0.5)
        loss = 0
        predictions = []
        for i in range(self.caption_len):
            captions_in = captions[:,i]
            word = self._word_embedding(captions_in,(i!=0))
            alpha = self._attend(h,norm_annotations,(i!=0))
            context = tf.matmul(tf.expand_dims(alpha,1),norm_annotations)
            context = tf.reshape(context,[self.batch_size,self.dim_emb])#[400000,512]
            _,(c,h) = lstm(inputs=tf.concat([context,word],1),state=(c,h))
            logits = self._decode_lstm(word,h,context,(i!=0))
            loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = captions_in, logits = logits))
            prob = tf.nn.softmax(logits)
            prediction = tf.argmax(prob,1)
            predict_word = list(word_to_idx.items())[prediction][0]
            predictions.append(predict_word)
        loss = loss/self.batch_size
        return loss,predictions
    def test(self,test_anno):
        self.test = True
        test_feat_dim = np.shape(norm_annotations)
        norm_annotations = tf.contrib.layers.batch_norm(test_anno)
        c,h = self._get_initial_input()
        lstm = tf.nn.rnn_cell.BasicLSTMCell(num_units = self.dim_hidden)
        lstm = tf.nn.rnn_cell.DropoutWrapper(lstm,
                input_keep_prob = 0.5,
                output_keep_prob = 0.5,
                state_keep_prob = 0.5)
        loss = 0
        predictions = []
        for i in range(self.caption_len):
            word = self._word_embedding(np.ones(test_feat_dim,1),True)
            alpha = self._attend(h,norm_annotations,True)
            context = tf.matmul(tf.expand_dims(alpha,1),norm_annotations)
            context = tf.reshape(context,[test_feat_dim,self.dim_emb])#[400000,512]
            _,(c,h) = lstm(inputs=tf.concat([context,word if i == 0 else logits],1),state=(c,h))
            logits = self._decode_lstm(word if i == 0 else logits,h,context,True)
            prob = tf.nn.softmax(logits)
            prediction = tf.argmax(prob,1)
            predict_word = list(word_to_idx.items())[prediction][0]
            predictions.append(predict_word)
        return predictions
    def optimizer(self):
        optimizer = tf.train.AdamOptimizer(
                    learning_rate = 0.0001,
                    beta1 = 0.9,
                    beta2 = 0.999,
                    epsilon = 0.000001
                    )
        return optimizer

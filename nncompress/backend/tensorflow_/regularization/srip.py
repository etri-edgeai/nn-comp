from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import backend as K

def l2_reg_ortho(model,
                 multiplier=1.0,
                 filter_=\
                    lambda x:x.__class__.__name__ == "Conv2D" or x.__class__.__name__ == "Dense"):
    ret = None
    cnt = 0
    for layer in model.layers:
        if filter_(layer):
            w = layer.get_weights()[0]
            if len(w.shape) == 4 and w.shape[0:2] != (1,1):
                continue
            cols = tf.size(w[0])
            rows = w.shape[0]
            w_col = tf.reshape(w, (-1, cols))
            w_t = tf.transpose(w_col, perm=[1,0])
            w_ = tf.linalg.matmul(w_t, w_col)
            I = tf.eye(cols, cols)
            
            w_tmp = (w_ - I)
            H = w_tmp.shape[0]
            u = tf.linalg.normalize(tf.random.normal((H,1)), ord=2, axis=0)[0]
            v = tf.linalg.normalize(tf.linalg.matmul(tf.transpose(w_t, perm=[1,0]), u), ord=2, axis=0)[0]
            u = tf.linalg.normalize(tf.linalg.matmul(w_t, v), ord=2, axis=0)[0]
            sigma = tf.tensordot(u, tf.linalg.matmul(w_t, v), axes=[0,0])
            
            if ret is None:
                ret = (sigma) ** 2
            else:
                ret = ret + (sigma) ** 2
            cnt += 1
    model.add_loss(lambda :multiplier * (ret / cnt))

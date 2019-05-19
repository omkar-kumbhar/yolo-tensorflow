from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
import numpy as np 
import pandas as pd
import os
from utils import DataLoader 

def conv_relu(inputs,filters,k_size,stride,padding,scope_name):
    """
    Method for convolution + leaky_relu on inputs

    :param inputs:
    :param filters:
    :param k_size:
    :param stride:
    :param padding:
    :param scope_name:
    """
    with tf.variable_scope(scope_name,reuse=tf.AUTO_REUSE) as scope:
        in_channels = inputs.shape[-1]
        kernel = tf.get_variable('kernel',
                                [k_size,k_size,in_channels,filters],
                                initializer=tf.truncated_normal_initializer())
        biases = tf.get_variable('biases',
                                [filters],
                                initializer=tf.random_normal_initializer())
        conv = tf.nn.conv2d(inputs,kernel,strides=[1,stride,stride,1],padding=padding)
        print('Layer  {} : Type = Conv, Size = {} x {}, Stride = {}, Filters = {}'.format(
              scope_name,k_size,k_size,stride,filters))

    return tf.nn.leaky_relu(conv+biases,name=scope.name)

def max_pool(inputs,k_size,stride,padding,scope_name='pool'):
    with tf.variable_scope(scope_name,reuse=tf.AUTO_REUSE) as scope:
        pool = tf.nn.max_pool(inputs,
                             ksize=[1,k_size,k_size,1],
                             strides=[1,stride,stride,1],
                             padding=padding)
        print('Layer  {} : Type = Conv, Size = {} x {}, Stride = {}, Padding = {}'.format(
              scope_name,k_size,k_size,stride,padding))

    return pool

def fully_connected(inputs,out_dim,scope_name='fc'):
    with tf.variable_scope(scope_name,reuse=tf.AUTO_REUSE) as scope:
        in_dim = inputs.shape[-1]
        w = tf.get_variable('weights',
                           [in_dim,out_dim],
                           initializer = tf.truncated_normal_initializer())
        b = tf.get_variable('bias',
                           [out_dim],
                           initializer = tf.constant_initializer(0.0))
        out = tf.matmul(inputs,w) + b
        return out

class YoloModel:
    def __init__(self,input_=None,output_dimensions=10):
        h = input_.shape[0]
        w = input_.shape[1]
        c = input_.shape[2]
        self.input_ = tf.placeholder(tf.float32,shape=[None,h,w,c])
        self.output_dimensions = tf.transpose(output_dimensions)#tf.placeholder(tf.float32,shape=[output_dimensions])

    def build_model(self):

        # layer1
        conv_1 = conv_relu(inputs=self.input_,filters=64,k_size=7,stride=2,
                           padding='SAME',scope_name="conv_1")
        max_pool1 = max_pool(inputs=conv_1,k_size=2,stride=2,
                             padding='SAME',scope_name='pool_1')
        
        # layer2
        conv_2 = conv_relu(inputs=max_pool1,filters=192,k_size=3,stride=1,
                           padding='SAME',scope_name="conv_2")
        max_pool2 = max_pool(inputs=conv_2,k_size=2,stride=2,
                             padding='SAME',scope_name='pool_2')
        
        #layer3 
        conv_3_1 = conv_relu(inputs=max_pool2,filters=128,k_size=1,stride=1,
                           padding='SAME',scope_name="conv_3_1")
        conv_3_2 = conv_relu(inputs=conv_3_1,filters=256,k_size=3,stride=1,
                           padding='SAME',scope_name="conv_3_2")
        conv_3_3 = conv_relu(inputs=conv_3_2,filters=256,k_size=1,stride=1,
                           padding='SAME',scope_name="conv_3_3")
        conv_3_4 = conv_relu(inputs=conv_3_3,filters=512,k_size=3,stride=1,
                           padding='SAME',scope_name="conv_3_4")
        max_pool3 = max_pool(inputs=conv_3_4,k_size=2,stride=2,
                             padding='SAME',scope_name='pool_3')
            
        # layer4
        # 4 times {1x1x256 + 3x3x512}
        conv_4_1 = conv_relu(inputs=max_pool3,filters=256,k_size=1,stride=1,
                           padding='SAME',scope_name="conv_4_1")
        conv_4_2 = conv_relu(inputs=conv_4_1,filters=512,k_size=3,stride=1,
                           padding='SAME',scope_name="conv_4_2")
        conv_4_3 = conv_relu(inputs=conv_4_2,filters=256,k_size=1,stride=1,
                           padding='SAME',scope_name="conv_4_3")
        conv_4_4 = conv_relu(inputs=conv_4_3,filters=512,k_size=3,stride=1,
                           padding='SAME',scope_name="conv_4_4")
        conv_4_5 = conv_relu(inputs=conv_4_4,filters=256,k_size=1,stride=1,
                           padding='SAME',scope_name="conv_4_5")
        conv_4_6 = conv_relu(inputs=conv_4_5,filters=512,k_size=3,stride=1,
                           padding='SAME',scope_name="conv_4_6")
        conv_4_7 = conv_relu(inputs=conv_4_6,filters=256,k_size=1,stride=1,
                           padding='SAME',scope_name="conv_4_7")
        conv_4_8 = conv_relu(inputs=conv_4_7,filters=512,k_size=3,stride=1,
                           padding='SAME',scope_name="conv_4_8")
        # 1x1x512
        conv_4_9 = conv_relu(inputs=conv_4_8,filters=512,k_size=1,stride=1,
                           padding='SAME',scope_name="conv_4_9")
        # 3x3x1024
        conv_4_10 = conv_relu(inputs=conv_4_9,filters=1024,k_size=3,stride=1,
                           padding='SAME',scope_name="conv_4_10")
        max_pool4 = max_pool(inputs=conv_4_10,k_size=2,stride=2,
                             padding='SAME',scope_name='pool_4')
        
        # layer 5
        # 2 times {1x1x512 + 3x3x1024}
        conv_5_1 = conv_relu(inputs=max_pool4,filters=512,k_size=1,stride=1,
                           padding='SAME',scope_name="conv_5_1")
        conv_5_2 = conv_relu(inputs=conv_5_1,filters=1024,k_size=3,stride=1,
                           padding='SAME',scope_name="conv_5_2")
        conv_5_3 = conv_relu(inputs=conv_5_2,filters=512,k_size=1,stride=1,
                           padding='SAME',scope_name="conv_5_3")
        conv_5_4 = conv_relu(inputs=conv_5_3,filters=1024,k_size=3,stride=1,
                           padding='SAME',scope_name="conv_5_4")
        # 3x3x1024
        conv_5_5 = conv_relu(inputs=conv_5_4,filters=1024,k_size=3,stride=1,
                           padding='SAME',scope_name="conv_5_5")
        # 3x3x1024-s2
        conv_5_6 = conv_relu(inputs=conv_5_5,filters=1024,k_size=3,stride=2,
                           padding='SAME',scope_name="conv_5_6")

        # layer 6
        # 3x3x1024
        conv_6_1 = conv_relu(inputs=conv_5_6,filters=1024,k_size=3,stride=1,
                           padding='SAME',scope_name="conv_6_1")
        # 3x3x1024
        conv_6_2 = conv_relu(inputs=conv_6_1,filters=1024,k_size=3,stride=1,
                           padding='SAME',scope_name="conv_6_2")

        # fully connected layers
        fc1 = fully_connected(inputs=conv_6_2,out_dim=self.output_dimensions,scope_name='fc1')
        #fc2 = fully_connected(inputs=fc1,out_dim=something,scope_name='fc2')
        return fc1
        
if __name__ == '__main__':
    
    # dl = DataLoader('./data/train.csv')
    # X_train, y_train = dl.preprocess_mnist()
    # # test = X_train[1]
    test = resize_image = np.zeros(shape=(448,448,1)).astype(np.float32)

    yolo = YoloModel(input_=test)
    yolo.build_model()
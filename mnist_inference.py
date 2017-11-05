# -*- coding:utf-8 -*-
import tensorflow as tf
#MNIST数据集相关的常数
INPUT_NODE = 784		#输入层的节点数
OUTPUT_NODE = 52		#输出层的节点数
LAYER1_NODE = 500		#配置神经网络的参数，隐藏层的节点数


#获取神经网络边上的权重，并将其加入名称为losses的集合中
def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None: tf.add_to_collection('losses', regularizer(weights))
    return weights

#定义一个辅助函数，给定神经网络的输入和正则参数，计算神经网络的前向传播过程
def inference(input_tensor, regularizer):
    #定义第一层神经网络的变量和前向传播过程
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
	#计算隐藏层的前向传播结果，这里使用了Relu激活函数
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    ##定义第二层神经网络的变量和前向传播过程
    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
	#计算输出层的前向传播结果
        layer2 = tf.matmul(layer1, weights) + biases

    #返回最后的前向传播结果
    return layer2

# -*- coding:utf-8 -*-
import tensorflow as tf
import mnist_inference
import mnist_input
import os
import numpy as np	
LEARNING_RATE_BASE = 0.8	#基础的学习率
LEARNING_RATE_DECAY = 0.99	#学习率的衰减率
REGULARIZATION_RATE = 0.0001	#描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 30000		#训练轮数
MOVING_AVERAGE_DECAY = 0.99	#滑动平均衰减率
MODEL_SAVE_PATH="MNIST_model/"
MODEL_NAME="mnist_model"



#训练模型的过程
def train():

    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
    #添加一个新的占位符用于输入正确值
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
    
    #计算正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    #计算当前参数下神经网络前向传播的结果
    y = mnist_inference.inference(x, regularizer)

    #一般会将代表训练轮数的变量指定为不可训练的参数
    global_step = tf.Variable(0, trainable=False)


    #给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    #在所有代表神经网络参数的变量上使用滑动平均
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    
    #计算交叉熵作为刻画预测值和真实值之间差距的损失函数，当分类问题只有一个正确答案时，
    #sparse_softmax_cross_entropy_with_logits函数可以加速交叉熵的计算
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    #计算当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
   
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
   
    #设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,			#基础的学习率，随着迭代的进行，更新变量时使用的学习率在这个基础上递减
        global_step,				#当前迭代的轮数
        52 ,	#过完所有的训练数据需要的迭代次数#######********* 
        LEARNING_RATE_DECAY,			#学习率衰减速度
        staircase=True)

    #使用tf.train.GradientDescentOptimizer优化算法来优化损失函数   包含了交叉熵损失和L2正则化损失
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    
    #在训练神经网络模型时，每过一遍数据既需要通过反向传播来更新神经网络中的参数，又需要更新每一个参数的滑动平均值。
    #为了一次完成多个操作，tensorflow提供了tf.control_dependencies机制
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    
    saver = tf.train.Saver(max_to_keep=1)
    
    #初始化会话并开始训练过程
    with tf.Session() as sess:
        start=0
        #实现断点续训，从最近保存的一次恢复数据
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            start = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        else:
            tf.initialize_all_variables().run()

	#迭代的训练神经网络
        for i in range(start,TRAINING_STEPS):   
    
	    #产生这一轮使用的一个batch的训练数据，并运行训练过程

            xs, ys = mnist_input.pre_process(i%52 +1)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})

            if i % 500 == 0 or i == 9999:
            	#输出当前的训练情况，通过损失函数的大小大概了解训练情况
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                #保存当前模型
                saver.save(sess, 'MNIST_model/my-model', global_step=global_step)


#主程序入口
def main(argv=None):
    #声明处理MNIST数据集的类，这个类在初始化时会自动下载数据
    train()

#tensorflow提供的一个主程序入口，tf.app.run会调用上面的main函数
if __name__ == '__main__':
    tf.app.run()



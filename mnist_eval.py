#coding:utf-8
import time
import tensorflow as tf
import mnist_inference
import mnist_train
#每10秒加载一次最新的模型
#加载的时间间隔
EVAL_INTERVAL_SECS = 10

def evaluate(mnist):

    #生成并维护一个计算图作为默认的计算图
    with tf.Graph().as_default() as g: 

	    #定义placeholder作为存放数据的地方（类型，维度，名称）
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')	
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

	    ############***********验证数据，在神经网络训练过程中通过验证数据来判断停止的条件和判断训练的效果
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        y = mnist_inference.inference(x, None)

        #检验使用了滑动平均模型的神经网络前向传播结果是否正确，tf.argmax(y, 1)计算每一个样例的预测答案，
        #tf.equal判断两个张量的每一维是否相等
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

	    #先将一个布尔型的数值转换为实数型，然后计算平均值，该平均值就是模型在这组数据上的正确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        #定义滑动平均模型使模型的泛化性更优
        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()

	    #声明tf.train.Saver类用于保存模型
        saver = tf.train.Saver(variables_to_restore)

        while True:
            #创建一个会话，并通过python的上下文管理器来管理这个会话
            with tf.Session() as sess:

                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)

                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    result = sess.run(y, feed_dict={x: #########********})	#测试数据集
                    code = ""
                    for i in result:
                        temp = list(i)
                        code += str(temp.index(max(temp)))
                        print 'input nums:'+nums[0:nums.index('.')]+'  recongize:'+code
 
                        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                        accuracy_score = sess.run(accuracy, feed_dict=validate_feed)########*********
                        print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)
#主程序
def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)	#载入MNIST数据
    evaluate(mnist)

if __name__ == '__main__':
    main()
"""
Extracting ../../../datasets/MNIST_data/train-images-idx3-ubyte.gz
Extracting ../../../datasets/MNIST_data/train-labels-idx1-ubyte.gz
Extracting ../../../datasets/MNIST_data/t10k-images-idx3-ubyte.gz
Extracting ../../../datasets/MNIST_data/t10k-labels-idx1-ubyte.gz
After 4001 training step(s), validation accuracy = 0.9844
After 5001 training step(s), validation accuracy = 0.9852
After 6001 training step(s), validation accuracy = 0.985
After 7001 training step(s), validation accuracy = 0.986
After 8001 training step(s), validation accuracy = 0.9844
After 9001 training step(s), validation accuracy = 0.986
After 10001 training step(s), validation accuracy = 0.9852
After 11001 training step(s), validation accuracy = 0.9842
After 12001 training step(s), validation accuracy = 0.9854
After 13001 training step(s), validation accuracy = 0.9856
After 14001 training step(s), validation accuracy = 0.9852
After 15001 training step(s), validation accuracy = 0.9852
After 16001 training step(s), validation accuracy = 0.9846
After 18001 training step(s), validation accuracy = 0.985
After 19001 training step(s), validation accuracy = 0.9852
After 20001 training step(s), validation accuracy = 0.9852
After 21001 training step(s), validation accuracy = 0.9856
After 22001 training step(s), validation accuracy = 0.9846
After 23001 training step(s), validation accuracy = 0.9854
"""

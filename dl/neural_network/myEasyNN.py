# @Time    : 2018/6/12 13:06
# @Author  : cap
# @FileName: myEasyNN.py
# @Software: PyCharm Community Edition

import pickle
import time

import numpy as np
import scipy.special as ss


class NeuralNetwork:
    """创建三层神经网络,输入层-隐藏层-输出层"""
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate, wih=None, who=None):
        """
        :param input_nodes: 输入层节点数
        :param hidden_nodes: 隐藏层节点数
        :param output_nodes: 输出层节点数
        :param learning_rate: 学习率
        :param wih:方便载入保存的训练好的数据
        :param who:...
        连接输入层和隐藏层的权重：(h_nodes) X (i_nodes) 维的矩阵
        连接隐藏层和输出层的权重：(o_nodes) X (h_nodes) 维的矩阵
        定义激活函数  activation_function
        """

        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes
        self.l_r = learning_rate

        # 创建随机正太分布的初始权重矩阵，均值为0，方差为隐藏层节点数的二次方根的倒数。
        # 为了避免出现饱和神经网络的情况，即考虑到输出值是输入值和权重值的点积，为了避免出现一个大的输出值，
        # 需要设置一个小的初始权重值，通常情况下我们会将范围设置为：正负连接数的倒数。这里采用效果更好的正太分布。
        # 注意维度设置
        if wih is None:
            self.wih = np.random.normal(0.0, pow(self.h_nodes, -0.5),
                                        (self.h_nodes, self.i_nodes))
        else:
            self.wih = wih
        if who is None:
            self.who = np.random.normal(0.0, pow(self.h_nodes, -0.5),
                                        (self.o_nodes, self.h_nodes))
        else:
            self.who = who
        # 激活函数$sigmoid(x) = \frac{1}{1 + e^{-x}}$
        # self.activation_function = lambda x: ss.expit(x)

    @staticmethod
    def activation_function(x):
        return ss.expit(x)

    # 训练方法
    def train(self, input_list, target_list):
        """
        :param input_list: 输入值，(i_nodes)x(1)维矩阵
        :param target_list: 目标值，(o_nodes)x(1)维
        通过正向输入处理得出最终输出，计算输出误差，计算隐藏层误差，通过梯度下降更新权重矩阵
        """
        inputs = np.array(input_list).reshape(-1, 1)
        targets = np.array(target_list).reshape(-1, 1)

        # x = w •  input
        # output = sigmoid(x)
        hidden_input = np.dot(self.wih, inputs)
        hidden_output = self.activation_function(hidden_input)

        final_input = np.dot(self.who, hidden_output)
        final_output = self.activation_function(final_input)

        # output_error, hidden_error
        # hidden_error = transpose(w) • output_error
        final_error = targets - final_output
        hidden_error = np.dot(self.who.T, final_error)

        # ▽w= - (error) * output * (1 - output) • transpose(input)
        # new_w = old_w + l_r * ▽w
        self.who += self.l_r * np.dot((final_error * final_output * (1 - final_output)), np.transpose(hidden_output))
        self.wih += self.l_r * np.dot((hidden_error * hidden_output * (1 - hidden_output)), np.transpose(inputs))

    # 用于验证测试
    def query(self, inputs_list):
        """
        :param inputs_list: 输入样本
        :return: final_output: 预测输出值
        """
        inputs = np.array(inputs_list, ndmin=2).T

        hidden_input = np.dot(self.wih, inputs)
        hidden_output = self.activation_function(hidden_input)

        final_input = np.dot(self.who, hidden_output)
        final_output = self.activation_function(final_input)
        return final_output


def train_model(n, filename, output_nodes):
    with open(filename, 'r') as f:
        while True:
            samples = f.readline()
            if samples:
                sample = samples.split(',')
                # 对输入值进行缩放，为防止出现0值情况，偏移0.01
                inputs = np.asfarray(sample[1:]) / 255 * 0.99 + 0.01
                # target的取值范围[0.01, 0.99],长度为output_nodes，取值为0.99的下表即是目标值
                target = np.zeros(output_nodes) + 0.01
                target[int(sample[0])] = 0.99
                n.train(inputs, target)
            else:
                break


def model_predict(n, filename):
    result_record = []
    with open(filename, 'r') as t:
        while True:
            samples = t.readline()
            if samples:
                sample = samples.split(',')
                test_inputs = np.asfarray(sample[1:]) / 255 * 0.99 + 0.01
                correct = int(sample[0])
                predict_list = n.query(test_inputs)
                index = np.argmax(predict_list)
                # print('correct: ', correct, '\tpredict: ', index)
                if correct == index:
                    result_record.append(1)
                else:
                    result_record.append(0)
            else:
                break
    return result_record


def show_result(start_time, result_record):
    # print('Record:\t', result_record)
    per = sum(result_record) / len(result_record) * 100
    print('Performance:\t%.2f%%' % per)

    end_time = time.time()
    print('Timing:\t%ds' % (end_time - start_time))


def save_model(n):
    with open('myEasyNN.params', 'wb') as f:
        params = (n.i_nodes, n.h_nodes, n.o_nodes, n.l_r, n.wih, n.who)
        pickle.dump(params, f)
    print('训练参数成功保存到当前文件夹中，用loda_model()获取。')


def load_model(filename):
    with open(filename, 'rb') as f:
        params = pickle.load(f)
        n = NeuralNetwork(*params)
    return n


def main():
    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10
    learning_rate = 0.3

    start_time = time.time()

    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    # http://www.pjreddie.com/media/files/mnist_train.csv
    # http://www.pjreddie.com/media/files/mnist_test.csv

    # small train
    # train = './data/mnist_train_100.csv'
    # test = './data/mnist_test_10.csv'
    train = '../../../data/ml/mnist/mnist_train.csv'
    test = '../../../data/ml/mnist/mnist_test.csv'

    train_model(n, train, output_nodes)
    result_record = model_predict(n, test)
    show_result(start_time, result_record)

    save_model(n)


if __name__ == '__main__':
    main()
'''load_model()使用方法
import myEasyNN

small_test = './data/mnist_test_10.csv'
n = myEasyNN.load_model('myEasyNN.params')
record = myEasyNN.model_predict(n,small_test)
myEasyNN.show_result(0, record)
'''

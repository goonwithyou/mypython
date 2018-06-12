# @Time    : 2018/6/12 13:06
# @Author  : cap
# @FileName: myEasyNN.py
# @Software: PyCharm Community Edition

import numpy as np
import scipy.special as ss
import time


class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5),
                                    (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5),
                                    (self.onodes, self.hnodes))

        self.lr = learningrate

        self.activation_function = lambda x: ss.expit(x)

    def train(self, input_list, target_list):
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T

        hidden_input = np.dot(self.wih, inputs)
        hidden_output = self.activation_function(hidden_input)

        final_input = np.dot(self.who, hidden_output)
        final_output = self.activation_function(final_input)

        final_error = targets - final_output
        hidden_error = np.dot(self.who.T, final_error)

        self.who += self.lr * np.dot((final_error * final_output * (1 - final_output)), np.transpose(hidden_output))
        self.wih += self.lr * np.dot((hidden_error * hidden_output * (1 - hidden_output)), np.transpose(inputs))

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T

        hidden_input = np.dot(self.wih, inputs)
        hidden_output = self.activation_function(hidden_input)

        final_input = np.dot(self.who, hidden_output)
        final_output = self.activation_function(final_input)
        return final_output


def main():
    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10

    learning_rate = 0.3

    start_time = time.time()

    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    # http://www.pjreddie.com/media/files/mnist_train.csv
    # http://www.pjreddie.com/media/files/mnist_test.csv
    with open('./data/mnist_train_100.csv', 'r') as f:
        data_list = f.readlines()

    # train
    for record in data_list:
        all_values = record.split(',')
        inputs = np.asfarray(all_values[1:]) / 255 * 0.99 + 0.01
        target = np.zeros(output_nodes) + 0.01
        target[int(all_values[0])] = 0.99
        n.train(inputs, target)

    # test
    result_record = []
    with open('./data/mnist_test_10.csv', 'r') as t:
        test_list = t.readlines()

    for test in test_list:
        test_values = test.split(',')
        test_inputs = np.asfarray(test_values[1:]) / 255 * 0.99 + 0.01
        correct = int(test_values[0])
        predict_list = n.query(test_inputs)
        index = np.argmax(predict_list)
        print('correct: ', correct, '\tpredict: ', index)
        if correct == index:
            result_record.append(1)
        else:
            result_record.append(0)

    print('Record:\t', result_record)
    per = sum(result_record) / len(result_record) * 100
    print('Performance:\t%.2f%%' % per)

    end_time = time.time()
    print('Timing:\t%d' % (end_time - start_time))


if __name__ == '__main__':
    main()

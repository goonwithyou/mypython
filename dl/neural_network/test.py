# @Time    : 2018/6/13 11:33
# @Author  : cap
# @FileName: test.py.py
# @Software: PyCharm Community Edition

# with open('../../../data/ml/mnist/mnist_train.csv', 'r') as f:
# with open('../../../data/ml/mnist/mnist_train.csv', 'r') as f:
#     print('ddd')
import time
import myEasyNN

start_time = time.time()
small_test = './data/mnist_test_10.csv'
n = myEasyNN.load_model('myEasyNN.params')
record = myEasyNN.model_predict(n,small_test)
myEasyNN.show_result(start_time, record)


# @Time    : 2018/7/1 17:56
# @Author  : cap
# @FileName: myNLTKGender.py
# @Software: PyCharm Community Edition
# @introduction: 性别识别
import random
import numpy as np
import nltk.corpus as nc
import nltk.classify as cf

# 获取已有的姓名数据
male_names = nc.names.words('male.txt')
female_names = nc.names.words('female.txt')
print('male_names--shape:', len(male_names))
print('female_names--shape:', len(female_names))

# model，acs：精度
models, acs = [], []

# 测试选择几个字母的预测效果最好，分别训练选取姓名最后1-5个字母，
# 共五个模型，评判哪个模型效果最好
for n_letters in range(1, 6):
    data = []
    # 构建训练数据格式（feature, male）
    for male_name in male_names:
        feature = {'feature': male_name[-n_letters:].lower()}
        data.append((feature, 'male'))
    for female_name in female_names:
        feature = {'feature': female_name[-n_letters:].lower()}
        data.append((feature, 'female'))

    random.seed(7)
    random.shuffle(data)
    train_data = data[:int(len(data) / 2)]
    test_data = data[int(len(data) / 2):]

    # 贝叶斯分类器
    model = cf.NaiveBayesClassifier.train(train_data)
    ac = cf.accuracy(model, test_data)
    models.append(model)
    acs.append(ac)

best_index = np.array(acs).argmax()
best_letters = best_index + 1
best_model = models[best_index]
best_ac = acs[best_index]
print(best_letters, '%.2f%%' % round(best_ac * 100, 2))

# 预测
names = ['Leonardo', 'Amy', 'Sam', 'Tom', 'Katherina', 'Taylor', 'Susanne']

print(names)
genders = []
for name in names:
    feature = {'feature': name[-best_letters:]}
    gender = best_model.classify(feature)
    genders.append(gender)
print(genders)


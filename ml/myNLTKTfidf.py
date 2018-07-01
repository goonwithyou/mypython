# @Time    : 2018/7/1 17:13
# @Author  : cap
# @FileName: myNLTKTfidf.py
# @Software: PyCharm Community Edition
# @introduction: 文本分类
import sklearn.datasets as sd
import sklearn.feature_extraction.text as ft
import sklearn.naive_bayes as nb


cld = {'misc.forsale': 'SALSE',
       'rec.motorcycles': 'MOTORCYCLES',
       'rec.sport.baseball': 'BASEBALL',
       'sci.crypt': 'CRYPTOGRAPHY',
       'sci.space': 'SPACE'
       }

train = sd.fetch_20newsgroups(subset='train',
                              categories=cld.keys(),
                              shuffle=True,
                              random_state=7)

# 获取训练数据，每条训练数据为一句话
train_data = train.data
# 训练数据label
train_y = train.target
categories = train.target_names

print(len(train_data))
print(len(train_y))
print(len(categories))

# 生成词频矩阵
cv = ft.CountVectorizer()
train_tfmat = cv.fit_transform(train_data)
print(train_tfmat.shape)

# 词频矩阵处理
tf = ft.TfidfTransformer()
train_x = tf.fit_transform(train_tfmat)

# 贝叶斯训练
model = nb.MultinomialNB()
model.fit(train_x, train_y)

test_data = [
    'the curveballs of right handed prtchers tend to curve to the left',
    'Caesar cipher is an ancient form of encryption',
    'This two-wheeler is really good on slippery roads'
]
# 把测试数据生成词频矩阵
test_tfmat = cv.transform(test_data)
# 处理词频矩阵，数据归一化
test_x = tf.transform(test_tfmat)
# 预测
pred_test_y = model.predict(test_x)
for sentence, index in zip(test_data, pred_test_y):
    print(sentence, '-->', cld[categories[index]])


# @Time    : 2018/7/1 17:05
# @Author  : cap
# @FileName: myNLTKBow.py
# @Software: PyCharm Community Edition
# @introduction: 词袋模型
import nltk.tokenize as tk
import sklearn.feature_extraction.text as ft


doc = 'The brown dog is runing. ' \
      'The black dog is in the black room. Running in the room is forbidden.'

sentences = tk.sent_tokenize(doc)
print(sentences)

# 词频矩阵
cv = ft.CountVectorizer()
tfrmat = cv.fit_transform(sentences).toarray()
words = cv.get_feature_names()
print(words)

print(tfrmat)

# @Time    : 2018/7/1 16:43
# @Author  : cap
# @FileName: myNLTKLemma.py
# @Software: PyCharm Community Edition
# @introduction: 词形还原
import nltk.stem as ns


words = ['table', 'probably', 'wolves', 'playing', 'is', 'dog',
         'the', 'beaches', 'grounded', 'dreamt', 'envision'
         ]

lemmatizer = ns.WordNetLemmatizer()
print('======lemmatize====n======')
for word in words:
    lemma = lemmatizer.lemmatize(word, 'n')
    print(lemma)


print('======lemmatize====v======')
for word in words:
    lemma = lemmatizer.lemmatize(word, 'v')
    print(lemma)
# @Time    : 2018/7/1 16:32
# @Author  : cap
# @FileName: myNLTKStem.py
# @Software: PyCharm Community Edition
# @introduction: 词干提取

import nltk.stem.porter as pt
import nltk.stem.lancaster as lc
import nltk.stem.snowball as sb


words = ['table', 'probably', 'wolves', 'playing', 'is', 'dog',
         'the', 'beaches', 'grounded', 'dreamt', 'envision'
         ]

# 词干提取
print('=======PorterStemmer=========')
stemmer = pt.PorterStemmer()
for word in words:
    stem = stemmer.stem(word)
    print(stem)

print('=======LancasterStemmer=========')
stemmer = lc.LancasterStemmer()
for word in words:
    stem = stemmer.stem(word)
    print(stem)

print('=======LancasterStemmer=========')
stemmer = sb.SnowballStemmer()
for word in words:
    stem = stemmer.stem(word)
    print(stem)
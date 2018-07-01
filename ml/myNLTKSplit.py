# @Time    : 2018/7/1 16:49
# @Author  : cap
# @FileName: myNLTKSplit.py
# @Software: PyCharm Community Edition
# @introduction:词块划分，指定块大小
import nltk.corpus as nc

doc = ' '.join(nc.brown.words()[:310])
print(doc)
words = doc.split()
chunks = []
for word in words:
    if len(chunks) == 0 or len(chunks[-1]) == 5:
        chunks.append([])
    chunks[-1].append(word)

for chunk in chunks:
    for word in chunk:
        print('{:15}'.format(word), end='')
    print()
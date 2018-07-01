# @Time    : 2018/7/1 18:31
# @Author  : cap
# @FileName: myNLTKSentment.py
# @Software: PyCharm Community Edition
# @introduction: 情感分析
import nltk.corpus as nc
import nltk.classify as cf
import nltk.classify.util as cu


# 读取正面影评信息
pdata = []
fileids = nc.movie_reviews.fileids('pos')
# print(fileids)
# 循环文件列表
for fileid in fileids:
    feature = {}
    # 获取文件中评论信息
    words = nc.movie_reviews.words(fileid)
    # 获取该条影评出现过的所有单词
    for word in words:
        feature[word] = True

    pdata.append((feature, 'POSITIVE'))
print('pdata:', len(pdata))
# 负面影评
ndata = []
fileids = nc.movie_reviews.fileids('neg')
# print(len(fileids))
for fileid in fileids:
    feature = {}
    words = nc.movie_reviews.words(fileid)
    for word in words:
        feature[word] = True

    ndata.append((feature, 'NEGATIVE'))
print('ndata:', len(ndata))
pnumb, nnumb = int(0.8 * len(pdata)), int(0.8 * len(ndata))

train_data = pdata[: pnumb] + ndata[: nnumb]
test_data = pdata[pnumb:] + ndata[nnumb:]

model = cf.NaiveBayesClassifier.train(train_data)
ac = cu.accuracy(model, test_data)
print('%.2f%%' % round(ac * 100, 2))
tops = model.most_informative_features()
for top in tops[:10]:
    print(top[0])

reviews = [
    'It is an amazing movie.',
    'This is a dull movie.I would never recommend it to anyone.',
    'The cinematography is pretty grate in this movie.',
    'This direction was terrible and the story was all over the place.'
]

sents, probs = [], []
for review in reviews:
    feature = {}
    words = review.split()
    for word in words:
        feature[word] = True
    pcls = model.prob_classify(feature)

    sent  = pcls.max()
    prob = pcls.prob(sent)
    sents.append(sent)
    probs.append(prob)
for review, sent, prob in zip(reviews,sents, probs):
    print(review, '->', sent, 'prob:', prob)
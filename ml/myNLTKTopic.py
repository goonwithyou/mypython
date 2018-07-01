# @Time    : 2018/7/2 6:21
# @Author  : cap
# @FileName: myNLTKTopic.py
# @Software: PyCharm Community Edition
# @introduction:
import nltk.tokenize as tk
import nltk.corpus as nc
import nltk.stem.snowball as sb
import gensim.models.ldamodel as gm
import gensim.corpora as gc

doc =  []
with open('./data/topic.txt', 'r') as f:
    for line in f.readlines():
        doc.append(line[:-1])

# 分词器
tokenizer = tk.RegexpTokenizer(r'\w+')
# 常用副词
stopwords = nc.stopwords.words('english')
# 词干提取
stemmer = sb.SnowballStemmer('english')
lines_tokens = []
for line in doc:
    tokens = tokenizer.tokenize(line.lower())
    line_tokens = []
    for token in tokens:
        if token not in stopwords:
            token = stemmer.stem(token)
            line_tokens.append(token)
    lines_tokens.append(line_tokens)

# 创建单词字典
dic = gc.Dictionary(lines_tokens)
bow = []
for line_tokens in lines_tokens:
    # 词袋矩阵
    row = dic.doc2bow(line_tokens)
    bow.append(row)
print(bow)

n_topics = 2
# Latent Dirichlet Allocation用于主题建模
model = gm.LdaModel(bow, num_topics=n_topics, id2word=dic, passes=25)
topics = model.print_topics(num_topics=n_topics, num_words=4)
print(topics)
# @Time    : 2018/7/1 11:25
# @Author  : cap
# @FileName: myNLTKToken.py
# @Software: PyCharm Community Edition
# @introduction:
import nltk as tk

text = 'Python is an easy to learn, powerful programming language. ' \
       'It has efficient high-level data structures and a simple but effective ' \
       'approach to object-oriented programming. ' \
       'Python’s elegant syntax and dynamic typing, ' \
       'together with its interpreted nature, make it an ideal language for scripting and ' \
       'rapid application development in many areas on most platforms.'

print(text)
print('====sent_tokenize====' * 3)

# 句分词器
tokens = tk.sent_tokenize(text)
for i in tokens:
    print(i)

print('====word_tokenize====' * 3)
# 词分类器
tokens = tk.word_tokenize(text)
for i in tokens:
    print(i)

print('====WordPunctTokenizer====' * 3)
# 词分类器2
tokenizer = tk.WordPunctTokenizer()
tokens = tokenizer.tokenize(text)
for i in tokens:
    print(i)

import nltk
import nltk_utils
x = "Hello. I am there for a reason."

wordtok = nltk.word_tokenize(x)
print(wordtok)
sentok = nltk.sent_tokenize(x)
print(sentok)

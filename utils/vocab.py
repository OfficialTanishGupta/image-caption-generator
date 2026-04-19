import nltk
from collections import Counter

class Vocabulary:
    def __init__(self):
        self.word2idx = {"<pad>":0, "<sos>":1, "<eos>":2, "<unk>":3}
        self.idx2word = {v:k for k,v in self.word2idx.items()}
        self.freq_threshold = 5

    def build_vocab(self, sentence_list):
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:
            for word in nltk.tokenize.word_tokenize(sentence.lower()):
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
                    idx += 1
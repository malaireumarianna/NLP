
e = 0.0001
lambdas = [0.25, 0.25, 0.25,0.25]
'''lambdas[3]=0.25
lambdas[1]=0.25
lambdas[2]=0.25
lambdas[0] = 1- lambdas[3] - lambdas[2] - lambdas[1]'''

while True:
    c_l0 = 0
    c_l1 = 0
    c_l2 = 0
    c_l3 = 0


    # train lambdas using the heldoutData
    for w in trigram_cz_heldout_data:
        # calculate expected counts for each lambda using uniform, unigrams, bigrams, trigrams
        c_l0 += lambdas[0] * uniformProbConditional(vocab_cz_train_data_size) / smoothedProbConditional(w[-1], w[1], w[0], lambdas, unigram_cz_train_data, bigram_cz_train_data, trigram_cz_train_data , vocab_cz_train_data_size)
        c_l1 += lambdas[1] * unigramProbConditional(w[-1], unigram_cz_train_data) / smoothedProbConditional(w[-1], w[1], w[0], lambdas, unigram_cz_train_data, bigram_cz_train_data, trigram_cz_train_data , vocab_cz_train_data_size)
        c_l2 += lambdas[2] * bigramProbConditional(w[-1], w[1], unigram_cz_train_data, bigram_cz_train_data, vocab_cz_train_data_size) / smoothedProbConditional(w[-1], w[1], w[0], lambdas, unigram_cz_train_data, bigram_cz_train_data, trigram_cz_train_data,  vocab_cz_train_data_size)
        c_l3 += lambdas[3] * trigramProbConditional(w[-1], w[1], w[0], unigram_cz_train_data, bigram_cz_train_data, trigram_cz_train_data,  vocab_cz_train_data_size) / smoothedProbConditional(w[-1], w[1], w[0], lambdas, unigram_cz_train_data, bigram_cz_train_data, trigram_cz_train_data,  vocab_cz_train_data_size)

    

    l0_new =  c_l0/(c_l0 + c_l1 + c_l2 +  c_l3)
    l1_new =  c_l1/(c_l0 + c_l1 + c_l2 +  c_l3)
    l2_new =  c_l2/(c_l0 + c_l1 + c_l2 +  c_l3)
    l3_new =  c_l3/(c_l0 + c_l1 + c_l2 +  c_l3)

        
    if abs(l0_new - lambdas[0])<e and abs(l1_new - lambdas[1])<e and abs(l2_new - lambdas[2])<e and abs(l3_new - lambdas[3])<e:
        break
          #return smoothed_prob, init_l0, init_l1, init_l2, init_l3
         
        #if abs(l0_new - l0)<e and abs(l1_new - l1)<e and abs(l2_new - l2)<e and abs(l3_new - l3)<e:

    lambdas[3]=l3_new
    lambdas[1]=l1_new
    lambdas[2]=l2_new
    lambdas[0]=l0_new


print(lambdas)


def calculate_word_classes(text, mode, limit):
    tokens = read_txt(text, mode, limit)
    unigr_dict = get_unigrams(tokens)
    bigr_dict, N = get_bigrams(tokens, 1)
    print("Unigram count", len(unigr_dict), "sum", sum(unigr_dict.values()))
    print("Bigram count", len(bigr_dict), "sum", sum(bigr_dict.values()))
    classes = {}
    if mode == "w":
        cut = 10    # use limit 10 for building classes of words
    else:
        cut = 5     # use limit 5 for building classes of tags
    for word in unigr_dict:
        if unigr_dict[word] >= cut:
            classes[word] = word
    unigr_left, unigr_right = Counter(), Counter()
    for bigram, count in bigr_dict.items():
        unigr_left[bigram[0]] += count
        unigr_right[bigram[1]] += count
    L_min_loss = []
    MI = []
    num_classes = []
    while len(classes) > 1:
        if len(classes) == 15:
            #print("Members of 15 classes")
            #with open(text + "_" + mode + "_classes.txt", "w", encoding="utf-8") as w:
            classes_15 = "\\\\\n\\\\".join(classes.values())
        num_classes.append(len(classes))
        mi_dict = mi_sum(unigr_left, unigr_right, bigr_dict, N)
        MI.append(sum(mi_dict.values()))
        L_min = loss_count(list(classes.keys()), bigr_dict, mi_dict, unigr_dict, unigr_left, unigr_right, N)
        L_min_loss.append(L_min)
        classes, unigr_left, unigr_right, unigr_dict, bigr_dict = merge_classes(unigr_dict, unigr_left, unigr_right, bigr_dict, classes, L_min)
    return L_min_loss, classes_15, MI, num_classes

  
import copy
import nltk
import numpy as np
from collections import Counter, defaultdict
from nltk import str2tuple


"""Trigram model smoothing"""
class TriEM_model:
    """Class for handling probabilities"""
    def __init__(self, tags):
        """Getting uniform, unigram, bigram, trigram probs"""
        # Get unigram counts and probs
        self.unigr_counts = Counter(tags)
        self.unigr_probs = defaultdict(float)
        unigr_N = sum(self.unigr_counts.values())
        for entry in self.unigr_counts:
            self.unigr_probs[entry] = float(self.unigr_counts[entry]) / unigr_N

        # Get bigram counts and probs
        self.bigr_counts = Counter(nltk.bigrams(tags))
        self.bigr_probs = defaultdict(float)
        for entry in self.bigr_counts:
            self.bigr_probs[entry] = float(self.bigr_counts[entry]) / self.unigr_counts[entry[0]]

        # Get trigram counts and probs
        self.trigr_counts = Counter(nltk.trigrams(tags))
        self.trigr_probs = defaultdict(float)
        for entry in self.trigr_counts:
            self.trigr_probs[entry] = float(self.trigr_counts[entry]) / self.bigr_counts[(entry[0], entry[1])]

        # Get uniform probability
        self.unif_prob = 1. / len(self.unigr_counts)

    def get_probs(self, hist2, hist1, word):
        """Getting probabilities for a word"""
        # Get probability for a word. If not present in the data, prob == 0.0
        p1 = self.unigr_probs[word]
        p2 = self.bigr_probs[(hist1, word)]
        p3 = self.trigr_probs[(hist2, hist1, word)]
        # Assign uniform prob when history for a word is unknown
        if p2 == 0.:
            p2 = 1. / len(self.unigr_probs) if self.unigr_probs[hist1] == 0. else 0.
        if p3 == 0.:
            p3 = 1. / len(self.unigr_probs) if self.bigr_probs[(hist2, hist1)] == 0. else 0.
        return self.unif_prob, p1, p2, p3

    def EM(self, H):
        """Smoothing EM algorithm: obtain lambdas"""
        # Initialize probability dictionaries
        self.lambdas = [0.25, 0.25, 0.25, 0.25]    # initial lambdas
        expected_lambdas = [0., 0., 0., 0.]
        old_lambdas = [0., 0., 0., 0.]
        # While changes of lambdas are significant
        while all(el > 0.00001 for el in np.absolute(np.subtract(old_lambdas, self.lambdas))):
            old_lambdas = copy.deepcopy(self.lambdas)
            # Create histories
            hist1 = "###"
            hist2 = "###"
            for word in H:
                p0, p1, p2, p3 = self.get_probs(hist2, hist1, word)
                # Compute smoothed prob
                p_lambda = self.lambdas[0] * p0 + self.lambdas[1] * p1 + self.lambdas[2] * p2 + self.lambdas[3] * p3
                # Update histories
                hist2 = hist1
                hist1 = word
                # Compute expected counts of lambdas
                expected_lambdas[0] += (self.lambdas[0] * p0 / p_lambda)
                expected_lambdas[1] += (self.lambdas[1] * p1 / p_lambda)
                expected_lambdas[2] += (self.lambdas[2] * p2 / p_lambda)
                expected_lambdas[3] += (self.lambdas[3] * p3 / p_lambda)
            # Recompute lambdas
            self.lambdas = [el / sum(expected_lambdas) for el in expected_lambdas]

    def trigr_smooth(self, data):
        """Smoothing of the whole language model using computed lambdas"""
        p_smoothed = defaultdict(float)
        # Create histories
        hist1 = "###"
        hist2 = "###"
        for word in data:
            p0, p1, p2, p3 = self.get_probs(hist2, hist1, word)
            # Compute smoothed prob
            p_lambda = self.lambdas[0] * p0 + self.lambdas[1] * p1 + self.lambdas[2] * p2 + self.lambdas[3] * p3
            # Rewrite probabilities
            p_smoothed[(hist2, hist1, word)] = p_lambda
            # Update histories
            hist2 = hist1
            hist1 = word
        return p_smoothed

    def trans_probs(self, hist2, hist1, word):

        p0, p1, p2, p3 = self.get_probs(hist2, hist1, word)
        p_lambda = self.lambdas[0] * p0 + self.lambdas[1] * p1 + self.lambdas[2] * p2 + self.lambdas[3] * p3
        return p_lambda


"""Lexical model and unigram model smoothing"""
class Lexical_model_smooth:

    def __init__(self, data):
        words, tags = data[0], data[1]
        self.w_t_counts = Counter([(words[i], tags[i]) for i in range(len(words) - 1)])
        self.t_counts = Counter(tags)
        self.words, self.tags = list(set(words)), list(set(tags))
        self.a = 2 ** (-20)
        self.V = len(self.words) * len(self.tags)   # vocabulary size
        self.N = len(words)        # data size

    def get_probs(self, word, tag):
        if (word, tag) in self.w_t_counts:
            return self.w_t_counts[(word, tag)] / self.t_counts[tag]
        return 1. / self.V

    def lex_smooth(self, data):
        p_smoothed = defaultdict(float)
        for token in data:
            word, tag = str2tuple(token)
            p_smoothed[(word, tag)] = self.get_probs(word, tag)
        return p_smoothed

    def emis_probs(self, word, tag):
        return self.get_probs(word, tag)

    def emis_probs_BW(self, word, tag):
        if (tag, word) in self.w_t_counts:
            return self.w_t_counts[(tag, word)]
        return 1. / self.V


class Unigram_model_smooth(Lexical_model_smooth):
    def get_probs(self, word, tag):
        """Getting probabilities for a tag"""
        return (self.t_counts[tag] + self.a) / (self.N + self.a * self.V)

    def init_probs(self, word, tag):
        """Getting an initial probability for Viterbi decoding"""
        return self.get_probs(word, tag)


#Function which calculates cross entropy
import numpy as np
def cross_entropy(lambdas, trigram_en_test_data, dataset, unigram_en_train_data, bigram_en_train_data, trigram_en_train_data,  vocab_en_train_data_size):
    crossEntropy = 0
    for w in trigram_en_test_data:
        crossEntropy -= np.log2(smoothedProbConditional(w[-1], w[1], w[0], lambdas, unigram_en_train_data, bigram_en_train_data, trigram_en_train_data,  vocab_en_train_data_size))  
    return crossEntropy/len(dataset)

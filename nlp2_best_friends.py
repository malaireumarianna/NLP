
import operator
import math

import math
# import nltk
import operator
from collections import Counter




def get_unigrams(tokens):
    """Creating unigram dictionary"""
    unigr_dict = Counter(tokens)
    return unigr_dict


def get_bigrams(tokens, distance=50):
    """Creating bigrams for different distances and creating bigram dictionary"""
    # bigrams = nltk.bigrams(tokens)
    bigrams = []
    for cur_tok_id in range(0, len(tokens) - 1):
        for dist_tok_id in range(1 if distance == 1 else 2, distance + 1):
            if len(tokens) > (cur_tok_id + dist_tok_id):
                bigrams.append((tokens[cur_tok_id], tokens[cur_tok_id + dist_tok_id]))
    bigr_dict = Counter(bigrams)
    return bigr_dict, len(bigrams)




#Count number of N-grams
def countNgram(ngram_list):
    cgrams = {}
    for item in ngram_list:
        cgrams[item] = (cgrams[item]+1) if item in cgrams else 1
    return cgrams


#Cleate list of unigrams
def createUnigrams(tokens_list):
    unigrams = []
    for ix, item in enumerate(tokens_list[:]):
        unigrams.append((item))
    return unigrams


#Create list of bigrams for some distance = 2 or 1
#for distance = 2 pairs are: (1, 3); (2, 4); (3, 5); (4, 6)

def createBigrams(tokens_list, dist):
    bigrams = []
    for ix, item in enumerate(tokens_list[:]):
        for i in range(1 if dist == 1 else 2, dist+1):
            if(ix+i) > len(tokens_list)-1: break
            bigrams.append((item, tokens_list[i+ix]))
    return bigrams


#Calculate Pointwise Mutual Information for given unigram, bigram and minimal treshold of occurencies of unigrams
def PMI(unigram, bigram, treshold):
    Nbi = sum(bigram.values()) * 1.0
    Nu0 = sum(unigram.values()) * 1.0
    Nu1 = sum(unigram.values()) * 1.0

    pmi = {}
    for key in bigram.keys():
        if (unigram[key[0]] >= treshold and unigram[key[1]] >= treshold):
            pmi[key] = math.log(1.0 * (bigram[key] / Nbi) / ((unigram[key[0]] / Nu0) * (unigram[key[1]] / Nu1)),2)

    sorted_pmi = sorted(pmi.items(), key=operator.itemgetter(1), reverse = True)
    return sorted_pmi



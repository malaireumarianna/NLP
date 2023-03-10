

import operator
import math

from collections import Counter



def read_txt(filename, mode, limit):
    """Reading input file and dividing into two groups of tokens (limited and all)"""
    tokens = ["<s>"]
    with open(filename, encoding="iso8859_2") as f:
        for line in f:
            if mode == "w":
                tokens.append(line.strip().split("/")[0])
            else:
                tokens.append(line.strip().split("/")[1])
        return tokens[:limit + 1]


def mi(c_x, c_y, c_xy, N):
    """Counting MI using formuala on slide 127"""
    return (c_xy / N) * math.log((c_xy * N) / (c_x * c_y), 2) if (c_xy * N) / (c_x * c_y) > 0 else 0


def mi_sum(unigr_left, unigr_right, bigr_dict, N):
    """Summing MI for all bigrams in a text and writing each to a dictionary"""
    mi_dict = {}
    for bigram in bigr_dict:
        mi_dict[bigram] = mi(unigr_left[bigram[0]], unigr_right[bigram[1]], bigr_dict[bigram], N)
    return mi_dict


def check_key(key, dict):
    """Checking the presence in a dictionary"""
    return dict[key] if key in dict else 0


def calculate_sum(unigr_dict, bigr_dict, mi_dict):
    """Summation part, formula on slide 127"""
    sum_dict = {}
    for word in unigr_dict:
        sum_dict[word] = sum(mi_dict[bigram] for bigram in bigr_dict if bigram[0] == word) + \
                         sum(mi_dict[bigram] for bigram in bigr_dict if bigram[1] == word) \
                         - check_key((word, word), mi_dict)
    return sum_dict


def calculate_sub(sum_dict, mi_dict, word_a, word_b):
    """Subtraction part, formula on slide 127"""
    return sum_dict[word_a] + sum_dict[word_b] - \
        check_key((word_a, word_b), mi_dict) - check_key((word_b, word_a), mi_dict)


def calculate_sum_lr(word_a, word_b, unigr_dict, bigr_dict, word_list, posit, N):
    """Calculate sum over left positioned and right positioned word a and word b"""
    sum_lr_ab = 0
    c_ab = unigr_dict[word_a] + unigr_dict[word_b]
    for word in word_list:
        c_lr = float(unigr_dict[word])
        if not (word == word_a or word == word_b):
            if posit == 0:
                c_lr_ab = bigr_dict[(word, word_a)] + bigr_dict[(word, word_b)]
            else:
                c_lr_ab = bigr_dict[(word_a, word)] + bigr_dict[(word_b, word)]
            sum_lr_ab += mi(c_lr, c_ab, c_lr_ab, N)
    return sum_lr_ab, c_ab


def calculate_add(word_a, word_b, unigr_dict, unigr_left, unigr_right, bigr_dict, N):
    """Calculate add, formula on slide 128"""
    sum_l_ab, c_l_ab = calculate_sum_lr(word_a, word_b, unigr_dict, bigr_dict, unigr_left, 0, N)
    sum_r_ab, c_r_ab = calculate_sum_lr(word_a, word_b, unigr_dict, bigr_dict, unigr_right, 1, N)
    c_ab_ab = bigr_dict[(word_a, word_a)] + bigr_dict[(word_a, word_b)] + \
              bigr_dict[(word_b, word_a)] + bigr_dict[(word_b, word_b)]
    return mi(c_l_ab, c_r_ab, c_ab_ab, N) + sum_l_ab + sum_r_ab


def loss_count(classes, bigr_dict, mi_dict, unigr_dict, unigr_left, unigr_right, N):
    """Losses calculation, formula from slide 131, finding minimal loss"""
    L = {}
    L_min = ("", 1.)
    sum_dict = calculate_sum(unigr_dict, bigr_dict, mi_dict)
    for id_a in range(0, len(classes)):
        for id_b in range(id_a + 1, len(classes)):   # so that not to repeat
            word_a = classes[id_a]
            word_b = classes[id_b]
            sub = calculate_sub(sum_dict, mi_dict, word_a, word_b)
            add = calculate_add(word_a, word_b, unigr_dict, unigr_left, unigr_right, bigr_dict, N)
            L[(word_a, word_b)] = sub - add
            if L_min[1] > L[(word_a, word_b)]:
                L_min = ((word_a, word_b), L[(word_a, word_b)])
    return L_min


def merge_classes(unigr_dict, unigr_left, unigr_right, bigr_dict, classes, L_min):
    """Updating dictionaries of counts and history of class merges"""
    word_a = L_min[0][0]
    word_b = L_min[0][1]

    # Merge unigram counts
    unigr_dict[word_a] += unigr_dict[word_b]
    del unigr_dict[word_b]
    unigr_left[word_a] += unigr_left[word_b]
    del unigr_left[word_b]
    unigr_right[word_a] += unigr_right[word_b]
    del unigr_right[word_b]

    # Merge bigram counts
    for bigram in list(bigr_dict):
        if bigram[0] == word_b or bigram[1] == word_b:
            if bigram[0] == word_b:
                change_to_0 = word_a
                change_from_0 = word_b
            else:
                change_to_0 = bigram[0]
                change_from_0 = bigram[0]
            if bigram[1] == word_b:
                change_to_1 = word_a
                change_from_1 = word_b
            else:
                change_to_1 = bigram[1]
                change_from_1 = bigram[1]
            bigr_dict[(change_to_0, change_to_1)] += bigr_dict[(change_from_0, change_from_1)]
            del bigr_dict[(change_from_0, change_from_1)]

    # Merge classes
    classes[word_a] = classes[word_a] + " " + classes[word_b]
    del classes[word_b]

    return classes, unigr_left, unigr_right, unigr_dict, bigr_dict

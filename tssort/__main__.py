import sys, os

sys.path.append(os.path.dirname(__file__))
from pairwise_ranking import *

if __name__ == "__main__":
    comparisons, sentences = open(sys.argv[1]).readlines(), open(sys.argv[2]).readlines()
    sentences = [sentence.strip() for sentence in sentences]
    comparisons = [comparison.strip() for comparison in comparisons]
    m, c = mle(comparisons, sentences)
    best_rankings(m, c)

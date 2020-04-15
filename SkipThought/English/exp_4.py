"""
Experiment #4
Unsupervised Classification with BERT model
Parsing of text
"""
from sentence_transformers import SentenceTransformer
from rouge import Rouge
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
from nltk import ngrams
from collections import Counter
from extract import getGroundTruth
# from sklearn.decomposition import PCA
from numpy import array, mean, cov
from numpy.linalg import eig
from nltk.corpus import stopwords
import re


def generateParagraphs(full_text):
	pattern = re.compile(r'^\s*$')
	print(len(full_text.split("\n")))
	lines = [line for line in full_text.split("\n") if line.strip(' ') != '']
	text = "\n".join(lines)
	print(text)
	return text


def parseText():
	"""
	Returns the headings of whole text
	:return: void
	"""
	gt = getGroundTruth()
	for full_text, catch_phrases in gt[:1]:
		texts = [re.sub(r'^(\d+) (.*)', r'\2', text) for text in full_text]
		legal_text = " ".join(texts)
		generateParagraphs(legal_text)


if __name__ == "__main__":
	parseText()


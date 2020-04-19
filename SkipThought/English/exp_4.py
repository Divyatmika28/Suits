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

def generateParagraph(sentences):
	"""
	Finds the paragraphs and headings from the full text
	:param sentences: full text sentences
	:return:
	"""
	paragraphs = []
	new_para = ""
	headings = []
	quotes_stack = ['$']
	quotes_ch = ["\'", "\""]
	ctr = -1
	resolved_flag = True
	next_heading = True
	for sentence in sentences:
		ctr += 1
		text = re.sub(r'(\d+) (.*)', r'\2', sentence)
		text = re.sub(r'\[(\d+)\]', r'', text)
		text_l = len(text)
		lines = [line.strip(' ') for line in text.split("\n") if line.strip(' ') != '']
		for line in lines:
			if line[0] in quotes_ch:
				ch = line[0]
				if ch == quotes_stack[-1]:
					quotes_stack.pop()
				else:
					quotes_stack.append(ch)
			if len(line) > 5 and line[-1] in quotes_ch:
				ch = line[-1]
				if ch == quotes_stack[-1]:
					quotes_stack.pop()
				else:
					quotes_stack.append(ch)
			new_para = "{} {}".format(new_para, line)
			if (len(quotes_stack) == 1) and (not resolved_flag):
				paragraphs.append(new_para)
				new_para = ""
				next_heading = True
			elif next_heading:
				if line != '\'':
					headings.append(line)
					next_heading = False
			resolved_flag = (len(quotes_stack) == 1)
	paragraphs.append(new_para)
	return paragraphs, headings


def parseText():
	"""
	Returns the headings of whole text
	:return: void
	"""
	gt = getGroundTruth()
	for full_text, catch_phrases in gt[:100]:
		paragraphs, headings = generateParagraph(full_text)
		for heading in headings[:3]:
			print(heading)
		print("="*20)


if __name__ == "__main__":
	parseText()


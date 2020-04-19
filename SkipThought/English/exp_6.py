"""
Experiment #6
Classifying the legal text based on the last catch phrase
"""
from rouge import Rouge
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
from nltk import ngrams
from collections import Counter
from extract import getGroundTruth
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


def generateParagraphs(full_text):
	pattern = re.compile(r'^\s*$')
	lines = [line for line in full_text.split("\n") if line.strip(' ') != '']
	text = " ".join(lines)
	return text


def cleanText(full_text):
	"""
	Clean the full text for text classification
	:param full_text: Full text to be cleaned
	:return:
	"""
	tokens = word_tokenize(full_text)
	tokens = [w.lower() for w in tokens]
	table = str.maketrans('', '', string.punctuation)
	stripped = [w.translate(table) for w in tokens]
	words = [word for word in stripped if word.isalpha()]
	stop_words = set(stopwords.words('english'))
	words = [w for w in words if w not in stop_words]
	return " ".join(words)


def nbTrain(legal_dataset: list):
	"""
	Train data set with class
	:param legal_dataset: data set containing legal text and their legal text class
	:return: weight of each word
	"""
	print("Training classifier... ", end='')
	inp_text = [cleanText(full_text) for full_text, _ in legal_dataset]
	inp_classes = [legal_class for _, legal_class in legal_dataset]
	class_num = 0
	legal_class = {}
	for inp_class in inp_classes:
		if inp_class not in legal_class:
			legal_class[inp_class] = class_num
			class_num += 1
	y_train = [legal_class[inp_class] for inp_class in inp_classes]
	cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english')
	x_train_set = cv.fit_transform(inp_text)
	naive_bayes = MultinomialNB()
	naive_bayes.fit(x_train_set, y_train)
	print("Done")
	return naive_bayes, cv, legal_class


def main():
	"""
	Executes the entire pipeline of the code
	:return: void
	"""
	gt = getGroundTruth()
	model_sum, gt_sum = [], []
	doc_n = len(gt)
	nb_dataset = []
	for full_text, catch_phrases in gt[:500]:
		texts = [re.sub(r'^(\d+) (.*)', r'\2', text) for text in full_text]
		legal_text = " ".join(texts)
		legal_text = generateParagraphs(legal_text)
		legal_class = catch_phrases[-1]
		nb_dataset.append((legal_text, catch_phrases[-1],))
	class_model, cv, legal_classes = nbTrain(nb_dataset)
	for full_text, catch_phrases in gt[:20]:
		texts = [re.sub(r'^(\d+) (.*)', r'\2', text) for text in full_text]
		legal_text = " ".join(texts)
		legal_text = generateParagraphs(legal_text)
		legal_text = cleanText(legal_text)
		text_cv = cv.transform([legal_text])
		legal_class = class_model.predict(text_cv)
		gt_legal_class = catch_phrases[-1]
		print(legal_class[0], gt_legal_class, legal_class[0] == legal_classes[gt_legal_class])


if __name__ == "__main__":
	main()

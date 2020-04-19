"""
Experiment #3
Unsupervised Classification with BERT model
PCA of important embeddings based on
S. Agarwal, N. K. Singh and P. Meel, "Single-Document Summarization Using Sentence Embeddings and K-Means Clustering,"
2018 (ICACCCN)
Won't work because the features to sentence ration doesn't help
Analysis of text
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


class LetsNet:
	def __init__(self, embedding_sz=5):
		self.encoder_model = SentenceTransformer('bert-base-nli-mean-tokens')
		self.rouge = Rouge()
		self.cluster_n = 5
		self.embedding_sz = embedding_sz
		self.kmeans = KMeans(n_clusters=self.cluster_n)
		self.stop_words = set(stopwords.words('english'))

	def encode(self, sentences):
		sentence_embeddings = self.encoder_model.encode(sentences)
		return sentence_embeddings

	def getCentroidRepresentative(self, clusters, sentence_embeddings):
		centroids = []
		for idx in range(self.cluster_n):
			centroid_id = np.where(clusters.labels_ == idx)[0]
			centroids.append(np.mean(centroid_id))
		closest, _ = pairwise_distances_argmin_min(clusters.cluster_centers_, sentence_embeddings)
		ordering = sorted(range(self.cluster_n), key=lambda k: centroids[k])
		return closest, ordering

	def evaluate(self, model_sum, gt_sum):
		"""
		Gives rouge score
		:param model_sum: list of summaries returned by the model
		:param gt_sum: list of ground truth summary from catchphrases
		:return: ROUGE score
		"""
		return self.rouge.get_scores(model_sum, gt_sum, avg=True)

	def getSentenceSummary(self, sentences: list):
		"""
		Returns summary of sentence
		:param sentences: list of sentences
		:return: summary text
		"""
		sentence_enc = self.encode(sentences)
		clusters = self.kmeans.fit(sentence_enc)
		closest, ordering = self.getCentroidRepresentative(clusters, sentence_enc)
		summary = '.'.join([sentences[closest[idx]] for idx in ordering]).replace('\n', ' ')
		return summary

	def main(self):
		"""
		Executes the entire pipeline of the code
		:return: void
		"""
		gt = getGroundTruth()
		model_sum, gt_sum = [], []
		doc_n = len(gt)
		for doc_idx in range(20):
			print("{}/{}".format(doc_idx, doc_n))
			full_text, catch_phrases = gt[doc_idx]
			summary = self.getSentenceSummary(full_text)
			model_sum.append(summary)
			gt_sum.append(".".join(catch_phrases))
		print("ROUGE score: {}".format(self.evaluate(model_sum, gt_sum)))

	def getIntroductions(self):
		"""
		Returns the first catch phrase of every doc
		:return: void
		"""
		gt = getGroundTruth()
		intro_word_freq = {}
		for full_text, catch_phrases in gt[:500]:
			intro_words = catch_phrases[0].split(" ")
			for word in intro_words:
				if word not in self.stop_words:
					if word not in intro_word_freq:
						intro_word_freq[word] = 0
					intro_word_freq[word] += 1
		intro_words = [(word, freq) for word, freq in intro_word_freq.items()]
		intro_words.sort(key=lambda x: x[1], reverse=True)
		print(intro_words)

	def getConclusion(self):
		"""
		Returns the last catch phrase of every doc
		:return: void
		"""
		gt = getGroundTruth()
		conclusion_freq = {}
		for full_text, catch_phrases in gt[:500]:
			conclusion = catch_phrases[-1]
			if conclusion not in conclusion_freq:
				conclusion_freq[conclusion] = 0
			conclusion_freq[conclusion] += 1
		conclusions = [(word, freq) for word, freq in conclusion_freq.items()]
		conclusions.sort(key=lambda x: x[1], reverse=True)
		for conclusion, _ in conclusions:
			print(conclusion)

	def getHeadings(self):
		"""
		Returns the headings of whole text
		:return: void
		"""
		gt = getGroundTruth()
		pattern = re.compile(r'.+(\n )+\n.+')
		for full_text, catch_phrases in gt[:1]:
			print("".join(full_text))
			headings = []
			for sent in full_text:
				if pattern.search(sent) is not None:
					sent = re.sub(r'(\n( )*)+\n', r'\n', sent)
					headings.append(sent)
			print(len(headings))
			for heading in headings:
				print("============================")
				print(heading)


if __name__ == "__main__":
	obj = LetsNet()
	obj.getHeadings()


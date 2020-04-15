"""
Experiment #2
Unsupervised Classification with BERT model
PCA of important embeddings based on
S. Agarwal, N. K. Singh and P. Meel, "Single-Document Summarization Using Sentence Embeddings and K-Means Clustering,"
2018 (ICACCCN)
Won't work because the features to sentence ration doesn't help
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


class LetsNet:
	def __init__(self, embedding_sz=5):
		self.encoder_model = SentenceTransformer('bert-base-nli-mean-tokens')
		self.rouge = Rouge()
		self.cluster_n = 5
		self.embedding_sz = embedding_sz
		self.kmeans = KMeans(n_clusters=self.cluster_n)

	def encode(self, sentences):
		sentence_embeddings = self.encoder_model.encode(sentences)
		pca_embedding = [np.array([embed_i[idx] for idx in range(6)]) for embed_i in sentence_embeddings]
		# print(pca_embedding)
		# pca_embedding = np.copy(sentence_embeddings[0, 1, 2, 3, 4, 5])
		return pca_embedding

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


if __name__ == "__main__":
	obj = LetsNet()
	obj.main()


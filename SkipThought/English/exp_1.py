"""
Unsupervised Classification with BERT model
ROUGE score: {'rouge-1': {'f': 0.1085521858562239, 'p': 0.06458889179681311, 'r': 0.5188028662134316},
'rouge-2': {'f': 0.029273494156752018, 'p': 0.016853038823294662, 'r': 0.1615466969241862},
'rouge-l': {'f': 0.10980431834856438, 'p': 0.06633119150710505, 'r': 0.42908802704082677}}
"""
from sentence_transformers import SentenceTransformer
from rouge import Rouge
from sklearn.cluster import KMeans, DBSCAN
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
from nltk import ngrams
from collections import Counter
from extract import getGroundTruth


def centroid_representative(clusters, sentence_embeddings):
	closest, _ = pairwise_distances_argmin_min(clusters.cluster_centers_, sentence_embeddings)
	return closest


def cluster(embeddings, method="kmeans", minimum_samples=6):
	if method == "dbscan":
		clusters = DBSCAN(eps=0.3, min_samples=minimum_samples)
	else:
		kmeans = KMeans(n_clusters=minimum_samples)
		clusters = kmeans.fit(embeddings)
	return clusters


def mostCommonPhrase(summary):
	result = []
	most_common_phrase = ""
	max_freq = 1
	for n in range(10, 3, -1):
		phrases = []
		for token in ngrams(summary.split(), n):
			phrases.append(' '.join(token))
		phrase, freq = Counter(phrases).most_common(1)[0]
		if freq > max_freq:
			max_freq = freq
			# result.append((phrase, n))
			# print(phrase)
			most_common_phrase = phrase
			summary = summary.replace(phrase, '')
	return most_common_phrase


def getCatchPhrase(cluster, full_text):
	cluster_sent = {}
	catch_phrase = []
	summary = []
	sentence_n = len(full_text)
	for sentence_id in range(sentence_n):
		label = cluster.labels_[sentence_id]
		if label not in cluster_sent:
			cluster_sent[label] = []
		cluster_sent[label].append(full_text[sentence_id])
	for label in cluster_sent.keys():
		summary_label = " ".join(cluster_sent[label])
		catch_phrase.append(mostCommonPhrase(summary_label))
	return catch_phrase


def evaluate(model_sum, gt_sum):
	"""
	Gives rouge score
	:param model_sum: list of summaries returned by the model
	:param gt_sum: list of ground truth summary from catchphrases
	:return: ROUGE score
	"""
	rouge = Rouge()
	return rouge.get_scores(model_sum, gt_sum, avg=True)


def main():
	"""
	Executes the entire pipeline of the code
	:return: void
	"""
	gt = getGroundTruth()
	model_sum, gt_sum = [], []
	print("Fetching encoder model...", end=" ")
	enc_model = SentenceTransformer('bert-base-nli-mean-tokens')
	print("Done")
	for full_text, catch_phrases in gt[:20]:
		# Embed each sentence
		sentence_embeddings = enc_model.encode(full_text)
		# Cluster each embedding
		cluster_n = 11
		clusters = cluster(sentence_embeddings, minimum_samples=cluster_n)
		centroids = []
		for idx in range(cluster_n):
			centroid_id = np.where(clusters.labels_ == idx)[0]
			centroids.append(np.mean(centroid_id))

		# Select representative cluster
		closest, _ = pairwise_distances_argmin_min(clusters.cluster_centers_, sentence_embeddings)
		ordering = sorted(range(cluster_n), key=lambda k: centroids[k])

		summary = '.'.join([full_text[closest[idx]] for idx in ordering]).replace('\n', ' ')
		model_sum.append(summary)
		gt_sum.append(".".join(catch_phrases))
	print("ROUGE score: {}".format(evaluate(model_sum, gt_sum)))


if __name__ == "__main__":
	main()


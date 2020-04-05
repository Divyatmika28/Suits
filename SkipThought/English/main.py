"""
Main Pipeline of the code
"""
from sentence_transformers import SentenceTransformer
from extract import getGroundTruth
from rouge import Rouge
from sklearn.cluster import KMeans, DBSCAN
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min


def cluster(embeddings, method="kmeans", minimum_samples=6):
	if method == "kmeans":
		kmeans = KMeans(n_clusters=minimum_samples)
		clusters = kmeans.fit(embeddings)
	elif method == "dbscan":
		clusters = DBSCAN(eps=0.3, min_samples=minimum_samples)
	return clusters


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
	for full_text, catch_phrases in gt:
		# Embed each sentence
		sentence_embeddings = enc_model.encode(full_text)

		# Cluster each embedding
		cluster_n = 11
		clusters = cluster(sentence_embeddings, minimum_samples=cluster_n)
		cluster_dict = {}
		for sentence_idx in range(len(full_text)):
			label = clusters.labels_[sentence_idx]
			if label not in cluster_dict:
				cluster_dict[label] = []
			cluster_dict[label].append(full_text[sentence_idx])
		for label in range(cluster_n):
			print(cluster_dict[label])
			print()
			print()
		centroids = []
		for idx in range(cluster_n):
			centroid_id = np.where(clusters.labels_ == idx)[0]
			centroids.append(np.mean(centroid_id))

		# Select representative cluster
		closest, _ = pairwise_distances_argmin_min(clusters.cluster_centers_, sentence_embeddings)
		ordering = sorted(range(cluster_n), key=lambda k: centroids[k])
		print(ordering)
		summary = ' '.join([full_text[closest[idx]] for idx in ordering]).replace('\n', ' ')
		model_sum.append(summary)
		print([(full_text[closest[idx]], closest[idx]) for idx in ordering])
		print(summary)
		print(len(catch_phrases))
		print(".".join(catch_phrases))
		gt_sum.append(".".join(catch_phrases))
		break
	print("ROUGE score: {}".format(evaluate(model_sum, gt_sum)))


if __name__ == "__main__":
	main()

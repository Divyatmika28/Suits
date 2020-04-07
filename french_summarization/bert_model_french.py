"""
This module accepts list of legal documents in french.
The input txt files has the format legal_text + @summary + summary (generated from preprocessing.py).
It uses BERT multilingual model to generate sentence embeddings. And later we cluster the embeddings 
and perfrom extractive summarization. 
And calculates the rouge score between true_summary and the generated summary. 

"""


import os
from sentence_transformers import SentenceTransformer
from rouge import Rouge
from sklearn.cluster import KMeans, DBSCAN
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min

def evaluate(model_sum, gt_sum):
	"""
	Gives rouge score
	:param model_sum: list of summaries returned by the model
	:param gt_sum: list of ground truth summary 
	:return: ROUGE score
	"""
	rouge = Rouge()
	return rouge.get_scores(model_sum, gt_sum, avg=True)



input_path_cases= os.listdir('cleaned_files/')
model_sum = []
true_sum = []
embedder = SentenceTransformer('distiluse-base-multilingual-cased')
root = "cleaned_files"
for file in input_path_cases:
    file_path = os.path.join(root,file)
    if '.txt' in file_path:
        with open(file_path) as f:
            corpus = f.read()
        res = corpus.split("@summary ")
        text,corpus_summary = res[0],res[1]
        legal_corpus = text.split(" , ")
        sentence_embeddings = embedder.encode(legal_corpus)
        enc_embedding = sentence_embeddings
        n_clusters = int(np.ceil(len(enc_embedding)**0.5))
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans = kmeans.fit(enc_embedding)
        avg = []
        closest = []
        for j in range(n_clusters):
            idx = np.where(kmeans.labels_ == j)[0]
            avg.append(np.mean(idx))
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_,\
                                                   enc_embedding)
        ordering = sorted(range(n_clusters), key=lambda k: avg[k])
        gen_summary = ' '.join([legal_corpus[closest[idx]] for idx in ordering])
        model_sum.append(gen_summary)
        true_sum.append(corpus_summary)


print("ROUGE score: {}".format(evaluate(model_sum, true_sum)))

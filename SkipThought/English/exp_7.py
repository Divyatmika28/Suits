"""
Experiment #5
Semi-supervised Classification with BERT model
Naive Bayes to classify legal documents.

Won't work because the features to sentence ration doesn't help
ROUGE score: {'rouge-1': {'f': 0.15891543192411192, 'p': 0.10786929286940353, 'r': 0.48338261200812216},
'rouge-2': {'f': 0.044784748961483566, 'p': 0.029803569885793052, 'r': 0.1348487065826596},
'rouge-l': {'f': 0.1593023632206829, 'p': 0.10777504509543559, 'r': 0.43031225319205646}}
"""
from sentence_transformers import SentenceTransformer
from rouge import Rouge
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
from extract import getGroundTruth
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


class LetsNet:
	def __init__(self, embedding_sz=5):
		self.encoder_model = SentenceTransformer('bert-base-nli-mean-tokens')
		self.rouge = Rouge()
		self.cluster_n = 5
		self.embedding_sz = embedding_sz
		self.kmeans = KMeans(n_clusters=self.cluster_n)
		self.cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english')
		self.legal_classes = {}
		self.stop_words = set(stopwords.words('english'))
		self.naive_bayes_model = MultinomialNB()
		self.legal_class_list = []

	def cleanText(self, full_text):
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
		words = [w for w in words if w not in self.stop_words]
		return " ".join(words)

	def encode(self, sentences):
		sentence_embeddings = self.encoder_model.encode(sentences)
		features_n = len(sentence_embeddings[0])
		sentences_n = len(sentences)
		norm_embedding = [[embed_i[idx] for idx in range(features_n)] for embed_i in sentence_embeddings]
		for idx in range(features_n):
			features = [embed_i[idx] for embed_i in sentence_embeddings]
			min_feature_val = min(features)
			max_feature_val = max(features)
			range_feature_val = max_feature_val - min_feature_val
			for sent_idx in range(sentences_n):
				norm_embedding[sent_idx][idx] = (norm_embedding[sent_idx][idx]-min_feature_val)/range_feature_val
		pca_embedding = [np.array([norm_vec[idx] for idx in range(features_n)]) for norm_vec in norm_embedding]
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
		inp_text = self.preprocess(sentences)
		text_cv = self.cv.transform([inp_text])
		legal_class = self.naive_bayes_model.predict(text_cv)
		summary = "{}{}.".format(summary, self.legal_class_list[legal_class[0]])
		print(summary)
		return summary

	def generateParagraphs(self, full_text):
		"""
		Remove empty lines
		:param full_text:
		:return:
		"""
		pattern = re.compile(r'^\s*$')
		lines = [line for line in full_text.split("\n") if line.strip(' ') != '']
		text = " ".join(lines)
		return text

	def nbTrain(self, legal_dataset: list):
		"""
		Train data set via Naive Bayes
		:param legal_dataset: data set containing legal text and their legal text class
		:return:
		"""
		print("Training classifier... ", end='')
		inp_text = [full_text for full_text, _ in legal_dataset]
		inp_classes = [legal_class for _, legal_class in legal_dataset]
		class_num = 0
		for inp_class in inp_classes:
			if inp_class not in self.legal_classes:
				self.legal_classes[inp_class] = class_num
				self.legal_class_list.append(inp_class)
				class_num += 1
		y_train = [self.legal_classes[inp_class] for inp_class in inp_classes]
		x_train_set = self.cv.fit_transform(inp_text)
		self.naive_bayes_model.fit(x_train_set, y_train)

	def preprocess(self, sentences):
		"""
		Preprocess for classifying the text
		:return:
		"""
		text = " ".join([re.sub(r'^(\d+) (.*)', r'\2', text) for text in sentences])
		text = self.generateParagraphs(text)
		return self.cleanText(text)

	def train(self):
		"""
		Trains a classifier for the legal text
		:return:
		"""
		gt = getGroundTruth()
		nb_dataset = []
		for full_text, catch_phrases in gt[:500]:
			legal_text = self.preprocess(full_text)
			nb_dataset.append((legal_text, catch_phrases[-1],))
		self.nbTrain(nb_dataset)

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
	obj.train()
	obj.main()


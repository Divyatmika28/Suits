"""
Main Pipeline of the code
"""
from SkipThought.English.extract import getGroundTruth
from rouge import Rouge


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
	for full_text, catch_phrases in gt:
		# TODO encode full_text using skip thought/paragram phrase encoding
		# TODO clustering of encodings using K-Means
		# TODO select representative of each cluster using Extractive method
		model_sum.append("summary.")
		gt_sum.append(".".join(catch_phrases))
	print("ROUGE score: {}".format(evaluate(model_sum, gt_sum)))


if __name__ == "__main__":
	main()

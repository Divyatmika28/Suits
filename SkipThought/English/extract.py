"""
Extracts full text and catchphrases from the AUSTLII dataset
Here the catchphrases are the ground truth
"""
import os
import xml.etree.ElementTree as ET


def extract(filename):
	"""
	Fetches all the sentences and catch phrases of a file and stores in a (<full text>, <catch-phrase>) tuple
	:param filename: file path to extract from
	:return: (<full text>, <catch-phrase>) for the path
	"""
	# TODO fill how to traverse XML tree and get tags and values
	pass


def getGroundTruth():
	"""
	Fetches all the full texts and their catch phrases and stores in a (<full text>, <catch-phrase>) tuple
	:return: list of (<full text>, <catch-phrase>) tuples
	"""
	gt = []
	for _, _, files in os.walk("./fulltext/"):
		for filename in files:
			gt.append(extract("./fulltext/{}".format(filename)))
	return gt


if __name__ == "__main__":
	pass

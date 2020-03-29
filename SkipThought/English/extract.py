"""
Extracts full text and catchphrases from the AUSTLII dataset
Here the catchphrases are the ground truth
"""
import os
import xml.etree.ElementTree as ET
import re


def extract(filename):
	"""
	Fetches all the sentences and catch phrases of a file and stores in a (<full text>, <catch-phrase>) tuple
	:param filename: file path to extract from
	:return: (<full text>, <catch-phrases>) for the path
	"""
	file_content = clean(filename)
	root = ET.fromstring(file_content)
	catchphrase_subtree = root.find('catchphrases')
	catchphrases = []
	for catchphrase in catchphrase_subtree:
		catchphrases.append(catchphrase.text)
	sentence_subtree = root.find('sentences')
	full_text = []
	for sentence in sentence_subtree:
		full_text.append(sentence.text)
	return full_text, catchphrases


def clean(filename):
	"""
	Cleans the xml for processing
	1. The attribute error in catchphrase tag
	2. Escape characters
	:param filename: file path of file to clean
	:return:
	"""
	with open(filename) as fp:
		file_content = fp.read()
	file_content = re.sub(r'<catchphrase "id=([a-z][0-9]*)">', r'<catchphrase id="\1">', file_content)
	file_content = re.sub(r'&([a-zA-Z])([a-zA-Z]*);', r'\1', file_content)
	return file_content


def getGroundTruth():
	"""
	Fetches all the full texts and their catch phrases and stores in a (<full text>, <catch-phrase>) tuple
	:return: list of (<full text>, <catch-phrase>) tuples
	"""
	print("Fetching dataset...", end=" ")
	gt = []
	for _, _, files in os.walk("./fulltext/"):
		for filename in files:
			gt.append(extract("./fulltext/{}".format(filename)))
	print("Done")
	return gt


if __name__ == "__main__":
	getGroundTruth()



"""
Experiment #8
Thematic Segmentation
"""
from sentence_transformers import SentenceTransformer
from rouge import Rouge
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
from nltk import ngrams
from collections import Counter
from extract import getGroundTruth
from sklearn.decomposition import PCA
from numpy import array, mean, cov
from numpy.linalg import eig
from nltk.corpus import stopwords
import re

def generateParagraph(sentences):
    """
    Finds the paragraphs and headings from the full text
    :param sentences: full text sentences
    :return:
    """
    paragraphs = {}
    newPara = ''
    paragraphsUnderHeading = {}
    newParaCount = 0
    headings = []
    quotes_stack = ['$']
    quotes_ch = ["\'", "\""]
    ctr = 0
    newParaCtr = 0
    headingCtr = 0
    headingPara = False
    paragraphStart = False
    for sentence in sentences:
        ctr += 1
        text = re.findall(' \n \n(\d+) ', sentence) 
        #print(text)
        if len(text)==0:
            headingPara = False
            text = re.findall('(\d+) ', sentence) 
            if len(text)==0:
                newPara = newPara + ' ' + sentence.strip()
            else:
                y = sentence.split(text[0]+' ', 1)
                if y[0] == '':
                    paragraphStart = True
                    paragraphs[newParaCtr] = newPara
                    if len(headings)-1 in paragraphsUnderHeading:
                        paragraphsUnderHeading[len(headings)-1].append(newPara)
                    else:
                        paragraphsUnderHeading[len(headings)-1] = [newPara]
                    newParaCtr += 1
                    newPara = '' + y[1].strip()
                else:
                    text = re.findall('\n(\d+) ', sentence) 
                    if len(text)==0:
                        newPara = newPara + ' ' + sentence.strip()
                    else:
                        z = sentence.split('\n'+text[0]+' ', 1)
                        newPara = newPara + ' ' + z[0].strip()
                        paragraphs[newParaCtr] = newPara
                        if len(headings)-1 in paragraphsUnderHeading:
                            paragraphsUnderHeading[len(headings)-1].append(newPara)
                        else:
                            paragraphsUnderHeading[len(headings)-1] = [newPara]
                        newPara = '' +  z[1].strip()
                        newParaCtr += 1
        else:
            string = ' \n \n'+text[0]+' '
            x = sentence.split(string, 1)
            headingPara = True
            heading = x[0].strip()
            headings.append(heading)
            if newPara == '':
                newPara = newPara + x[1].strip(' ')
            else:
                paragraphs[newParaCtr] = newPara
                if len(headings)-2 in paragraphsUnderHeading:
                    paragraphsUnderHeading[len(headings)-2].append(newPara)
                else:
                    paragraphsUnderHeading[len(headings)-2] = [newPara]
                newPara = '' + x[1].strip(' ')
            newParaCtr += 1
            headingCtr += 1
    
    #print('Hola', headings)
    return paragraphs, headings, paragraphsUnderHeading

def thematicSegmentation(paragraphs, headings, paragraphsUnderHeading):
    """
    Returns the segments of each para
    :return: segements and the paragraphs under them
    """
    segments = {}
    if len(headings) == 1:
        segments['Analysis'] = paragraphsUnderHeading[0]
    elif len(headings) == 2:
        segments['Context'] = paragraphsUnderHeading[0]
        segments['Analysis'] = paragraphsUnderHeading[1]
    elif len(headings) == 3:
        segments['Context'] = paragraphsUnderHeading[1]
        segments['Analysis'] = paragraphsUnderHeading[2]
        segments['Introduction'] = paragraphsUnderHeading[0]
    elif len(headings) == 4:
        segments['Introduction'] = paragraphsUnderHeading[0]
        segments['Context'] = paragraphsUnderHeading[1]
        segments['Analysis'] = paragraphsUnderHeading[2]
        segments['Conclusion'] = paragraphsUnderHeading[3]
    else:
        segments['Introduction'] = paragraphsUnderHeading[0]
        segments['Conclusion'] = paragraphsUnderHeading[len(headings)-1]
        val = int(1+int(0.25*(len(headings)-2)))
        #print(val)
        for x in range(1,val):
            if 'Context' in segments:
                segments['Context'] + paragraphsUnderHeading[x]
            else:
                segments['Context'] = paragraphsUnderHeading[x]
        for x in range(val,len(headings)-1):
            if 'Analysis' in segments:
                segments['Analysis'] + paragraphsUnderHeading[x]
            else:
                segments['Analysis'] = paragraphsUnderHeading[x]
        
    return segments

def parseText():
    """
    Returns the headings of whole text
    :return: void
    """
    gt = getGroundTruth()
    paraSegmentsFinal = []
    for full_text, catch_phrases in gt[:100]:
        #print(full_text)
        paragraphs, headings, paragraphsUnderHeading = generateParagraph(full_text)
        for heading in headings:
            break
            #print(heading)
        #print(len(paragraphs), len(headings), len(paragraphsUnderHeading))
        paraSegments = thematicSegmentation(paragraphs, headings, paragraphsUnderHeading)
        paraSegmentsFinal.append(paraSegments)
    return paraSegmentsFinal
        #print(len(paraSegments['Analysis']),len(paraSegments['Context']),len(paraSegments['Introduction']),len(paraSegments['Conclusion']))

if __name__ == "__main__":
    parseText()

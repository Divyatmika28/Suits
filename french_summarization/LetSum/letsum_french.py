from collections import OrderedDict
import operator
import nltk

import formulated_constants_french

MAX_LENGTH_SUMMARY = 100
MAX_PERCENT_SUMMARY = 34

summary_division = {'Introduction': 10, 'Context': 25, 'Analysis': 60, 'Conclusion': 5, 'Citation': 0}

mapping_to_letsum = ['INTRODUCTION', 'CONTEXT', 'ANALYSIS', 'CONCLUSION', 'CITATION']

cue_phrases = formulated_constants_french.categorical_phrases


def tokennize_file(file):
	with open(file, 'r') as f:
		txt = f.read()
	case_txt = txt.split('@summary ')[0]
	case_txt = case_txt.replace(';', '.')
	case_txt = case_txt.splitlines()

	return case_txt


def print_num_paragraphs_assigned(num_paragraphs_assigned):
	for category, number in num_paragraphs_assigned.items():
		print(category, ': ', number)


def letsum(file):

	case_in_paragraph = tokennize_file(file)
	# used for checking purposes, not functionality of letsum
	num_paragraphs_assigned = {'Introduction': 0, 'Context': 0, 'Analysis': 0, 'Conclusion': 0, 'Citation': 0}

	if len(case_in_paragraph) > 4:
		category_txt = {'Introduction': '', 'Context': '', 'Analysis': '', 'Conclusion': '', 'Citation': ''}
		intro_to_context = False
		context_to_analysis = False
		analysis_to_conclusion = False
		remaning_is_conclusion = False
		# max number of paragaph a category could have based on percentage
		# ex. max number of paragraphs intro could have is 10% of number of paragaphs
		max_num_intro = int(len(case_in_paragraph) * .1)
		max_num_context = int(len(case_in_paragraph) * .25)
		max_num_analysis = int(len(case_in_paragraph) * .45)
		max_num_conclusion = int(len(case_in_paragraph) * .1)

		for num_paragraph in range(len(case_in_paragraph)):
			current_paragraph = case_in_paragraph[num_paragraph]
			paragraph_score = {'Introduction': 0, 'Context': 0, 'Analysis': 0, 'Conclusion': 0, 'Citation': 0}
			words = current_paragraph.split(' ')
			for category, key_phrase in cue_phrases.items():
				for word in words:
					if word in key_phrase:
						paragraph_score[category] += 1

			if num_paragraph == 0: #assume first paragraph will always be intro
				category_txt['Introduction'] += current_paragraph
				max_num_intro -= 1
				intro_to_context = True
				# print('Intro got paragaph 0')
				num_paragraphs_assigned['Introduction'] += 1
			if num_paragraph > 0 and intro_to_context:
				if paragraph_score['Introduction'] >= paragraph_score['Context'] and max_num_intro > 0:
					category_txt['Introduction'] += current_paragraph
					max_num_intro -= 1
					# print('Intro got paragaph ', num_paragraph)
					num_paragraphs_assigned['Introduction'] += 1
				else:
					category_txt['Context'] += current_paragraph
					max_num_context -= 1
					intro_to_context = False
					context_to_analysis = True
					# print('Context got paragaph ', num_paragraph)
					num_paragraphs_assigned['Context'] += 1
			if num_paragraph > 1 and context_to_analysis:
				if paragraph_score['Context'] >= paragraph_score['Analysis'] and max_num_context >0:
					category_txt['Context'] += current_paragraph
					max_num_context -= 1
					# print('Context got paragaph ', num_paragraph)
					num_paragraphs_assigned['Context'] += 1
				else:
					category_txt['Analysis'] += current_paragraph
					max_num_analysis -= 1
					context_to_analysis = False
					analysis_to_conclusion = True
					# print('Analysis got paragaph ', num_paragraph)
					num_paragraphs_assigned['Analysis'] += 1
			if num_paragraph > 2 and analysis_to_conclusion:
				if paragraph_score['Analysis'] >= paragraph_score['Conclusion'] and max_num_analysis > 0:
					category_txt['Analysis'] += current_paragraph
					max_num_analysis -= 1
					# print('Analysis got paragaph ', num_paragraph)
					num_paragraphs_assigned['Analysis'] += 1
				else:
					category_txt['Conclusion'] += current_paragraph
					analysis_to_conclusion = False
					remaning_is_conclusion = True
					# print('Conclusion got paragaph ', num_paragraph)
					num_paragraphs_assigned['Conclusion'] += 1
			if remaning_is_conclusion:
				category_txt['Conclusion'] += current_paragraph
				# print('Conclusion got paragaph ', num_paragraph)
				num_paragraphs_assigned['Analysis'] += 1

	if len(case_in_paragraph) == 4:
		category_txt = {'Introduction': '', 'Context': '', 'Analysis': '', 'Conclusion': '', 'Citation': ''}
		category_txt['Introduction'] += case_in_paragraph[0]
		category_txt['Context'] += case_in_paragraph[1]
		category_txt['Analysis'] += case_in_paragraph[2]
		category_txt['Conclusion'] += case_in_paragraph[3]

	# Less than 4 paragraphs, categorize the sentences.
	if len(case_in_paragraph) < 4:
		for num_paragraph in range(len(case_in_paragraph)):
			current_paragraph = case_in_paragraph[num_paragraph]
			sentences = nltk.tokenize.sent_tokenize(current_paragraph, language='french')
			for sentence in sentences:
				sentence_score = {'Introduction': 0, 'Context': 0, 'Analysis': 0, 'Conclusion': 0, 'Citation': 0}
				words = sentence.split(' ')
				for category, key_phrase in cue_phrases.items():
					for word in words:
						if word in key_phrase:
							sentence_score[category] += 1

				category_highest_score = max(sentence_score.items(), key=operator.itemgetter(1))[0]
				print('category_highest_score: ', category_highest_score)
				category_txt[category_highest_score] = sentence

	print(file)
	# print(len(case_in_paragraph))
	print_num_paragraphs_assigned(num_paragraphs_assigned)
	for cat, para in category_txt.items():
		print(cat, ': \n', para)


	return 'summary_txt'
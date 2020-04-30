import os, json
import letsum_french

def main():
	# for root, dirs, files in os.walk('../preprocessed_data/'):
	# for file in files:
	# 	summary = letsum_test.LetSum('../preprocessed_data/' + file)
	# 	with open(file.replace('.', '_letsum.'), 'w') as f:
	# 		f.write(summary)

	input_path = os.listdir('PATH') # dir of files you want to segemnt 
	root = 'PATH' # dir of files you want to segemnt 
	for file in input_path:
		file_path = os.path.join(root, file)
		segmentation = letsum_french.letsum(file_path)
		file_name = file.split('.')[0]
		with open('DIR/OF/OUTPUT/' + file_name + '_segment.txt', 'w') as f:
			json.dump(segmentation, f)

if __name__ == '__main__':
	main()
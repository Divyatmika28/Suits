import os
import letsum
import letsum_test

def main():
	# for root, dirs, files in os.walk('../preprocessed_data/'):
	# files = ['JURITEXT000041585787.txt', 'JURITEXT000038708903.txt', 'JURITEXT000041585788.txt', 'JURITEXT000038708904.txt',
	# 		 'JURITEXT000041585789.txt']
	# for file in files:
	# 	summary = letsum_test.LetSum('../preprocessed_data/' + file)
	# 	with open(file.replace('.', '_letsum.'), 'w') as f:
	# 		f.write(summary)

	summary = letsum_test.LetSum('../preprocessed_data/' + 'JURITEXT000006951557.txt')
	with open('../letsum_summary/test.txt', 'w') as f:
		f.write(summary)


if __name__ == '__main__':
	main()
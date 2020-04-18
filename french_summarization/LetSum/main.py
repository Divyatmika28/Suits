import os
import letsum_french

def main():
	# for root, dirs, files in os.walk('../preprocessed_data/'):
	# files = ['JURITEXT000041585787.txt', 'JURITEXT000038708903.txt', 'JURITEXT000041585788.txt', 'JURITEXT000038708904.txt',
	# 		 'JURITEXT000041585789.txt']
	# for file in files:
	# 	summary = letsum_test.LetSum('../preprocessed_data/' + file)
	# 	with open(file.replace('.', '_letsum.'), 'w') as f:
	# 		f.write(summary)

	files = [
	'JURITEXT000007013226.txt',
	'JURITEXT000032050184.txt',
	'JURITEXT000007013227.txt',
	'JURITEXT000032050187.txt',
	'JURITEXT000007015569.txt',
	'JURITEXT000035612444.txt',
	'JURITEXT000007015570.txt',
	'JURITEXT000035612496.txt',
	'JURITEXT000007015641.txt',
	'JURITEXT000035974653.txt',
	'JURITEXT000007015642.txt',
	'JURITEXT000035975500.txt'
	]
	for file in files:
		summary = letsum_french.letsum('../CASS-dataset/cleaned_files/' + file)
		with open('../letsum_summary/' + file + '_summary.txt', 'w') as f:
			f.write(summary)


if __name__ == '__main__':
	main()
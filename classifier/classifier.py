import os
import re

class Classifier:

	DATASET_PATH = "./data"
	TEST_DATASET_PATH = DATASET_PATH + "/test"
	TRAIN_DATASET_PATH = DATASET_PATH + "/train"

	SMOOTHING_DELTA = 0.5

	def __init__(self):
		self.vocabulary = []
		self.spam_vocabulary_frequencies = {}
		self.ham_vocabulary_frequencies = {}

		self.spam_vocabulary_probs = {}
		self.ham_vocabulary_probs = {}

	def build_model(self):
		all_training_file_names = os.listdir(Classifier.TRAIN_DATASET_PATH)

		for file_name in all_training_file_names:
			file = open(Classifier.TRAIN_DATASET_PATH+"/"+file_name, encoding="latin-1")
			lines = file.readlines()
			vocabulary_frequencies = self.spam_vocabulary_frequencies if 'spam' in file_name else self.ham_vocabulary_frequencies
			
			for line in lines:
				words_list = re.split('[^a-zA-Z]',line.lower())
				# remove empty strings
				words_list = [word for word in words_list if word]
				# populate vocabulary
				for word in words_list:
					# push data in vocabulary
					self.vocabulary.append(word)
					
					if word in vocabulary_frequencies:
						vocabulary_frequencies[word] += 1
					else:
						vocabulary_frequencies[word] = 1
			file.close()

		self.vocabulary = list(set(self.vocabulary))

	def add_smoothing(self):
		spam_words = self.spam_vocabulary_frequencies.keys()
		ham_words = self.ham_vocabulary_frequencies.keys()

		for word in self.vocabulary:
			if word not in spam_words:
				self.spam_vocabulary_frequencies[word] = Classifier.SMOOTHING_DELTA
			else:
				self.spam_vocabulary_frequencies[word] += Classifier.SMOOTHING_DELTA

			if word not in ham_words:
				self.ham_vocabulary_frequencies[word] = Classifier.SMOOTHING_DELTA
			else:
				self.ham_vocabulary_frequencies[word] += Classifier.SMOOTHING_DELTA

	def write_model_data(self):
		file = open('model.txt', "w")
		spam_total_words = sum(self.spam_vocabulary_frequencies.values())
		ham_total_words = sum(self.ham_vocabulary_frequencies.values())
		
		for index, word in enumerate(sorted(self.vocabulary)):
			index = int(index) + 1
			if index != 1:
				file.write("\n")
			file.write("%s  " % index)
			file.write(word + '  ')
			file.write("%s  " % (int(self.ham_vocabulary_frequencies[word] - Classifier.SMOOTHING_DELTA)))
			file.write("%s  " % (self.ham_vocabulary_frequencies[word]/ham_total_words))
			file.write("%s  " % (int(self.spam_vocabulary_frequencies[word] - Classifier.SMOOTHING_DELTA)))
			file.write("%s" % (self.spam_vocabulary_frequencies[word]/spam_total_words))

		file.close()
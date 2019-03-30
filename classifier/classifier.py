import os
import re
import math

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
			self.ham_vocabulary_probs[word] = self.ham_vocabulary_frequencies[word]/ham_total_words
			self.spam_vocabulary_probs[word] =self.spam_vocabulary_frequencies[word]/spam_total_words

			index = int(index) + 1
			if index != 1:
				file.write("\n")
			file.write("%s  " % index)
			file.write(word + '  ')
			file.write("%s  " % (int(self.ham_vocabulary_frequencies[word] - Classifier.SMOOTHING_DELTA)))
			file.write("%s  " % self.ham_vocabulary_probs[word])
			file.write("%s  " % (int(self.spam_vocabulary_frequencies[word] - Classifier.SMOOTHING_DELTA)))
			file.write("%s" % self.spam_vocabulary_probs[word])

		file.close()

	def test_model(self):
		file_to_write = open('baseline-result.txt', "w")
		all_training_file_names = os.listdir(Classifier.TEST_DATASET_PATH)
		
		total_test_file_count = len(all_training_file_names)

		for index, file_name in enumerate(all_training_file_names):
			# import pdb; pdb.set_trace()
			print("** file number %s/%s **" % (index, total_test_file_count))
			file = open(Classifier.TEST_DATASET_PATH+"/"+file_name, encoding="latin-1")
			lines = file.readlines()

			spam_score = 0
			ham_score = 0
			
			total_words = []
			
			for line in lines:
				words_list = re.split('[^a-zA-Z]',line.lower())
				# remove empty strings
				words_list = [word for word in words_list if word]
				total_words.extend(words_list)

			for word in total_words:
				#TODO: what to do when the word is not in train data?
				if(word in self.spam_vocabulary_probs.keys()):
					spam_score += math.log(self.spam_vocabulary_probs[word])
					ham_score += math.log(self.ham_vocabulary_probs[word])

			index = int(index) + 1
			if index != 1:
				file_to_write.write("\n")

			file_to_write.write("%s  " % index)
			file_to_write.write("%s  " % file_name)
			if spam_score >= ham_score:
				file_to_write.write("spam  ")
			else:
				file_to_write.write("ham  ")
			file_to_write.write("%s  " % ham_score)
			file_to_write.write("%s  " % spam_score)
			if 'ham' in file_name:
				correct_class = "ham"
				file_to_write.write("ham  ")
			else:
				correct_class = "spam"
				file_to_write.write("spam  ")

			if correct_class == "spam" and spam_score >= ham_score:
				file_to_write.write("right")
			elif correct_class == "ham" and ham_score > spam_score:
				file_to_write.write("right")
			else:
				file_to_write.write("wrong")

			file.close()
		
		file_to_write.close()
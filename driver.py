from classifier.classifier import Classifier

def main():
	classifier = Classifier()
	classifier.build_model()
	classifier.add_smoothing()
	classifier.spam_vocabulary_probs, classifier.ham_vocabulary_probs = classifier.write_model_data('model.txt', classifier.vocabulary)
	classifier.test_model('baseline-result.txt', classifier.spam_vocabulary_probs, classifier.ham_vocabulary_probs)
	print("------Experiment 2, Stop Words Filtering------")
	classifier.experiment2_stop_words()
	print("------Experiment 3, Word Length Filtering------")
	classifier.experiment3_length_filtering()
	print("------Experiment 4, Frequency 1 Filtering------")
	classifier.experiment4_frequency_filtering(file_name='frequencyFiltered0',lower_cutoff_frequency=1, higher_cutoff_frequency=1)
	print("------Experiment 4, Frequency <=5 Filtering------")
	classifier.experiment4_frequency_filtering(file_name='frequencyFiltered1',lower_cutoff_frequency=0, higher_cutoff_frequency=5)
	print("------Experiment 4, Frequency <=10 Filtering------")
	classifier.experiment4_frequency_filtering(file_name='frequencyFiltered2',lower_cutoff_frequency=0, higher_cutoff_frequency=10)
	print("------Experiment 4, Frequency <=15 Filtering------")
	classifier.experiment4_frequency_filtering(file_name='frequencyFiltered3',lower_cutoff_frequency=0, higher_cutoff_frequency=15)
	print("------Experiment 4, Frequency <=20 Filtering------")
	classifier.experiment4_frequency_filtering(file_name='frequencyFiltered4',lower_cutoff_frequency=0, higher_cutoff_frequency=20)
	print("------Experiment 4, Top 10 percent Filtering------")
	classifier.experiment4_most_frequent_filtering('mostFrequencyFiltered0', 10)
	print("------Experiment 4, Top 15 percent Filtering------")
	classifier.experiment4_most_frequent_filtering('mostFrequencyFiltered1', 15)
	print("------Experiment 4, Top 20 percent Filtering------")
	classifier.experiment4_most_frequent_filtering('mostFrequencyFiltered2', 20)
	print("------Experiment 4, Top 25 percent Filtering------")
	classifier.experiment4_most_frequent_filtering('mostFrequencyFiltered3', 25)

	experiment5_file_name = 'smoothing'
	
	for n in range(0,11):
		smoothing_value = round((n*0.1),1)
		file_name = experiment5_file_name+str(smoothing_value)
		print("------Experiment 5, smoothing value %s------"% smoothing_value)
		classifier_5 = Classifier()
		classifier_5.build_model()
		classifier_5.add_smoothing(smoothing_value)
		classifier_5.spam_vocabulary_probs, classifier_5.ham_vocabulary_probs = classifier_5.write_model_data(file_name+'model.txt', classifier_5.vocabulary, smoothing_value=smoothing_value)
		classifier_5.test_model(file_name+'baseline-result.txt', classifier_5.spam_vocabulary_probs, classifier_5.ham_vocabulary_probs)

main()
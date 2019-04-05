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
	print("------Experiment 4, Frequency Filtering------")
	classifier.experiment4_frequency_filtering(cutoff_frequency=5)
main()
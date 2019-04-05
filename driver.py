from classifier.classifier import Classifier

def main():
	classifier = Classifier()
	classifier.build_model()
	classifier.add_smoothing()
	classifier.spam_vocabulary_probs, classifier.ham_vocabulary_probs = classifier.write_model_data('model.txt', classifier.vocabulary)
	classifier.test_model('baseline-result.txt', classifier.spam_vocabulary_probs, classifier.ham_vocabulary_probs)
	classifier.experiment2_stop_words()
main()
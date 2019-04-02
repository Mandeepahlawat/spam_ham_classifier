from classifier.classifier import Classifier

def main():
	classifier = Classifier()
	classifier.build_model()
	classifier.add_smoothing()
	classifier.write_model_data()
	classifier.test_model()
	classifier.experiment2_stop_words()
main()
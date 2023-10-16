"""
This file contains the class FrameClassifier, which is used to predict the aspects for a given sentence.
Since the original classifier in https://github.com/Leibniz-HBI/argument-aspect-corpus-v1 is written in Flair but
we use PyTorch, we need to use this wrapper class to load the model and use it for prediction.
"""
from flair.data import Sentence
from flair.models import SequenceTagger


class FrameClassifier:
    def __init__(self, model_path: str):
        self.classifier = SequenceTagger.load(model_path)

    def get_list_of_labels_with_probas(self, sentence):
        self.classifier.predict(sentence)
        labels_lists = []
        for pair in sentence:
            if pair.get_labels():
                labels_lists.append((pair.get_labels()[0].value, pair.get_labels()[0].score))
            else:
                # other label
                labels_lists.append(("O", 1))
        return labels_lists

    def predict_aspect_labels_for_sentences(self, sentences):
        """
        Uses the classifier to predict the labels for each sentence.
        Returns a list of lists of tuples. Each list of tuples represents the labels for one sentence with the probability.
        """
        flair_sentences = [Sentence(sentence) for sentence in sentences]
        predicted_labels = [self.get_list_of_labels_with_probas(sentence) for sentence in flair_sentences]
        return predicted_labels

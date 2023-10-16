import json
import pathlib
from typing import List, Set, Tuple, Optional, Dict

import numpy
import torch

from Utils import get_label_saved_name
from embedding_script import main as create_embedding_pickles
from transformers import pipeline
from sentence_transformers import util
from rouge import Rouge
import pickle

from loguru import logger


def load_frame_embedding(label, config):
    def load_pickle(file_path):
        with open(file_path, 'rb') as f:
            embeddings = pickle.load(f)
        return embeddings

    embedding_type = config["data_manifold_closeness_embedding_type"]
    topic = config["dataset_name"]
    label_name = get_label_saved_name(label)
    embedding_file_path = f"sbert_embeddings/{config.get('s_bert_model_name', 'all-MiniLM-L6-v2')}/{embedding_type}/{topic}/{label_name}.pkl"
    try:
        return load_pickle(embedding_file_path)
    except FileNotFoundError:
        logger.opt(exception=True).warning("We have to calculate the embeddings for {} first", label)
        create_embedding_pickles(config)
        logger.debug("Done, now loading the calculated embeddings")
    return load_pickle(embedding_file_path)


def load_topic_embedding(config):
    embedding_file_path = f"sbert_embeddings/{config.get('s_bert_model_name', 'all-MiniLM-L6-v2')}/whole_topic/{config['dataset_name']}.pkl"
    with open(embedding_file_path, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings


def get_list_of_label_sets(predicted_labels: List[List[Tuple[str, float]]]) -> List[Set[Tuple[str, float]]]:
    """
    Get the set of labels from predicted_labels.
    :param predicted_labels: list of list of tuples (label, probability)
    :return: set of labels with corresponding probabilities
    """
    label_set_list = []
    for sentence in predicted_labels:
        labels_dict = {}
        for label in sentence:
            if label[0] in labels_dict.keys():
                labels_dict[label[0]] = max(labels_dict[label[0]], label[1])
            else:
                labels_dict[label[0]] = label[1]
        label_set = {(key, value) for key, value in labels_dict.items()}
        label_set_list.append(label_set)
    return label_set_list


def prediction_based_weighted_jaccard(predicted_labels: Set[Tuple[str, float]],
                                      true_labels: Set[str],
                                      other_label_list: List[str]) -> float:
    """
    Calculates the prediction based weighted jaccard similarity between two sets of labels.
    :param predicted_labels: list of tuples (label, probability)
    :param true_labels: set of labels
    :return: float representing the similarity
    """
    weighted_intersection = 0
    for label in predicted_labels:
        if label[0] in other_label_list:
            continue
        if any(map(lambda true: true.startswith(label[0]), true_labels)):
            weighted_intersection += label[1]

    weighted_union = len(true_labels)
    for label in predicted_labels:
        if label[0] in other_label_list:
            continue
        if all(map(lambda true: not true.startswith(label[0]), true_labels)):
            weighted_union += label[1]

    similarity = weighted_intersection / weighted_union
    return similarity


class EvaluationMetrics:
    def __init__(self, config):
        # Initialize any resources or settings needed for evaluation
        self.config = config

        logger.info("Loading required models for evaluation ({} config keys: {})",
                    len(self.config), ", ".join(self.config.keys()))

        logger.debug("... for grammar score")
        grammar_model = self.config.get("grammar_model", "textattack/roberta-base-CoLA")
        logger.info("OK, loading grammar checker model \"{}\". Some hints: \"textattack/roberta-base-CoLA\" might be "
                    "not so accurate, but smooth, while \"yiiino/deberta-v3-base-cola\" is larger and much more "
                    "sharper (might be sometimes overfitting)", grammar_model)
        self.grammar_checker = pipeline(
            task="text-classification",
            model=grammar_model,
            tokenizer=grammar_model,
            framework="pt",
            model_kwargs={"id2label": {0: "unacceptable", 1: "acceptable"}},
            binary_output=False,
            return_all_scores=False,
            device="cuda:0" if torch.cuda.is_available() else "cpu"
        )
        logger.debug("Loaded grammar checker model: {}", self.grammar_checker)

        logger.debug("... for sparsity score")
        self.rouge_recall = Rouge(metrics=["rouge-1", "rouge-l"], stats=["r"])
        logger.debug("Loaded rouge model: {}", self.rouge_recall)

    def check_validity(self,
                       target_labels_set: Set[str],
                       predicted_labels_list: List[Set[Tuple[str, float]]]) -> List[float]:
        """
        Compute the validity scores based on label_set and target_labels.
        :param target_labels_set: set of labels that should be the result label set of the counterfactuals
        :param predicted_labels_list: list of predicted labels for each counterfactual as list of tuples (label, probability)
        :return: list of frame set similarities
        """
        validity_scores = []
        for i in range(len(predicted_labels_list)):
            similarity = prediction_based_weighted_jaccard(predicted_labels_list[i], target_labels_set,
                                                           self.config["other_labels"])
            validity_scores.append(similarity)
        return validity_scores

    def check_proximity(self, original_sentence: str, counterfactual_sentences: List[str],
                        sbert_model: torch.nn.Module) -> List[float]:
        sentence_proximity_list = []
        # Compute embedding for both lists
        original_embed = sbert_model.encode(original_sentence, convert_to_tensor=True)

        for sentence in counterfactual_sentences:
            counterfactual_embed = sbert_model.encode(sentence, convert_to_tensor=True)
            # Compute cosine-similarity
            cosine_similarity = float(util.cos_sim(original_embed, counterfactual_embed))
            sentence_proximity_list.append(cosine_similarity)
        return sentence_proximity_list

    def check_sparsity(self,
                       original_sentence: str,
                       counterfactual_sentences: List[str]) -> List[float]:
        """
        Calculate the sparsity of the counterfactual sentences compared to the original sentence.
        Sparsity is calculated via the rouge score and the average between rouge-1 and rouge-l is used.
        :param original_sentence: original sentence
        :param counterfactual_sentences: list of counterfactual sentences
        :return: list of rouge scores for each counterfactual sentence as float
        """
        rouge_scores = []
        for cf in counterfactual_sentences:
            # Use the average of the two rouge scores
            try:
                score_rouge = self.rouge_recall.get_scores(original_sentence, cf)[0]
                score_r1 = score_rouge["rouge-1"]["r"]  # recall
                score_rl = score_rouge["rouge-l"]["r"]  # recall
                avg_score = (score_r1 + score_rl) / 2
            except ValueError:
                logger.opt(exception=True).warning("Could not calculate rouge score for \"{}\" and \"{}\"",
                                                   original_sentence, cf)
                avg_score = 0
            rouge_scores.append(avg_score)
        return rouge_scores

    def check_diversity(self, counterfactual_sentences: List[str], sbert_model: torch.nn.Module) -> List[float]:
        """
        Method calculates diversity by taking the average of the similarities from one counterfactual to all the others and
        averaging this with the maximum observed similarity.
        """
        # Calculate similarity between one cf and the rest
        diversity_value_list = []
        for main_sentence in counterfactual_sentences:
            main_sentence_embed = sbert_model.encode(main_sentence, convert_to_tensor=True)
            max_similarity = 0
            tmp_similarity_list = []
            for other_sentence in counterfactual_sentences:
                if main_sentence == other_sentence:
                    continue
                other_sentence_embed = sbert_model.encode(other_sentence, convert_to_tensor=True)
                # Compute cosine-similarity
                cosine_similarity = float(util.cos_sim(main_sentence_embed, other_sentence_embed))
                tmp_similarity_list.append(cosine_similarity)
                # Save max value
                max_similarity = cosine_similarity if max_similarity < cosine_similarity else max_similarity
            if len(tmp_similarity_list) == 0:
                diversity_value_list.append(1)
            else:
                avg_similarity = sum(tmp_similarity_list) / len(tmp_similarity_list)
                avg_between_avg_and_max = (max_similarity + avg_similarity) / 2
                # 1 - avg_between_avg_and_max is the diversity score
                diversity_value_list.append(1 - avg_between_avg_and_max)
        return diversity_value_list

    def check_data_manifold_closeness_for_frame(self,
                                                counterfactual_sentences: List[str],
                                                target_labels: Set[str],
                                                sbert_model: torch.nn.Module) -> List[float]:
        """
        Use embedding space for specific frame and check if counterfactuals are close to it.
        """
        # Load embeddings for target labels and store them in a dictionary
        topic_to_embedding_dict = {}
        for label in target_labels:
            if label in self.config['other_labels']:
                continue
            frame_embedding = load_frame_embedding(label, self.config)
            topic_to_embedding_dict[label] = frame_embedding

        # Compute embedding per counterfactuals and compare to frame embeddings
        frame_closeness_list = []
        for cf in counterfactual_sentences:
            embedded_cf = sbert_model.encode(cf, convert_to_tensor=True)
            cf_similarities_per_frame = {}
            # Calculate average similarity to top k neirest neighbors per target label
            for label in target_labels:
                if label in self.config['other_labels']:
                    continue
                all_sentences_embedded = topic_to_embedding_dict[label]['embeddings']
                # Step 1: Calculate cosine similarity between cf and all frame embeddings
                sentence_similarities = []
                for sentence in all_sentences_embedded:
                    if isinstance(sentence, numpy.ndarray):
                        sentence = torch.from_numpy(sentence)
                    sentence_similarity = util.cos_sim(embedded_cf, sentence.to(embedded_cf.device))
                    sentence_similarities.append(sentence_similarity.item())

                # Sort sentences by similarity
                sorted_similatiries = sorted(sentence_similarities, reverse=True)
                # Get average of top k sentence similarities
                k = self.config['top_k_embedding_similarities']
                top_k_similarities = sorted_similatiries[:k]
                average_cosine_similarity = sum([similarity for similarity in top_k_similarities]) / k
                cf_similarities_per_frame[label] = average_cosine_similarity
            frame_closeness_list.append(cf_similarities_per_frame)
            # TODO: Do we want to average here?
        average_closeness_list = []
        for similarities_dict in frame_closeness_list:
            cf_closeness = 0
            for label, closeness in similarities_dict.items():
                cf_closeness += closeness
            average_closeness_list.append(cf_closeness / len(similarities_dict))
        return average_closeness_list

    def check_data_manifold_closeness_for_topic(self,
                                                counterfactual_sentences: List[str],
                                                sbert_model: torch.nn.Module) -> List[float]:
        topic_embedding = load_topic_embedding(self.config)["embeddings"]
        # Compute embedding per counterfactuals and compare to topic embedding
        topic_similarity_per_cf = []
        for cf in counterfactual_sentences:
            embedded_cf = sbert_model.encode(cf, convert_to_tensor=True)
            topic_closeness_list = []
            for sentence in topic_embedding:
                if isinstance(sentence, numpy.ndarray):
                    sentence = torch.from_numpy(sentence)
                sentence_similarity = util.cos_sim(embedded_cf, sentence.to(embedded_cf.device))
                topic_closeness_list.append(sentence_similarity.item())
            # sort by similarity
            topic_closeness_list.sort(reverse=True)
            # get average of top k sentence similarities
            k = self.config['top_k_embedding_similarities']
            top_k_similarities = topic_closeness_list[:k]
            average_cosine_similarity = sum(top_k_similarities) / k
            topic_similarity_per_cf.append(average_cosine_similarity)
        return topic_similarity_per_cf

    def check_grammar(self, counterfactual_sentences: List[str]) -> List[float]:
        logger.debug("OK, let's check the {} sentences for grammar using \"{}\"",
                     len(counterfactual_sentences), self.grammar_checker.model.config._name_or_path)

        grammar_scores = self.grammar_checker(counterfactual_sentences)
        logger.trace("Got grammar scores {}, have to convert now...", grammar_scores)

        return [gs.get("score", .5) if gs.get("label", "acceptable") == "acceptable" else 1-gs.get("score", .5)
                for gs in grammar_scores]

    def evaluate(self,
                 original_sentence_dict,
                 generated_sentences_list: List[List[str]],
                 classifier,
                 s_bert,
                 save_path: Optional[pathlib.Path] = None):
        """
        Evaluate counterfactuals based on the metrics.
        :param original_sentence_dict: dictionary containing the original sentence and its labels and target labels
        :param generated_sentences_list: list of list of generated counterfactual sentences
        """
        counterfactuals_evaluation = []
        for i, generated_sentences in enumerate(generated_sentences_list):
            predicted_labels = classifier.predict_aspect_labels_for_sentences(generated_sentences)
            list_of_predicted_label_sets = get_list_of_label_sets(predicted_labels)
            validity_scores = self.check_validity(original_sentence_dict["target_labels"][i],
                                                  list_of_predicted_label_sets)
            proximity_scores = self.check_proximity(original_sentence_dict["sentences"][i], generated_sentences, s_bert)
            sparsity_scores = self.check_sparsity(original_sentence_dict["sentences"][i], generated_sentences)
            diversity_scores = self.check_diversity(generated_sentences, s_bert)
            grammar_scores = self.check_grammar(generated_sentences)
            data_manifold_closeness_scores_per_label = self.check_data_manifold_closeness_for_frame(generated_sentences,
                                                                                                    original_sentence_dict[
                                                                                                        "target_labels"][
                                                                                                        i],
                                                                                                    s_bert)
            data_manifold_closeness_scores_per_topic = self.check_data_manifold_closeness_for_topic(generated_sentences,
                                                                                                    s_bert)
            # iterating counterfactuals and saving metric dict
            iner_dict_list = []
            for j, counterfactual in enumerate(generated_sentences):
                # Assuming you have a way to get metrics for each counterfactual sentence
                iner_dict_list.append({
                    'cf_string': counterfactual,
                    'pred_labels': str(predicted_labels[j]),
                    'pred_labels_set': str(list_of_predicted_label_sets[j]),
                    'validity': validity_scores[j],
                    'proximity': proximity_scores[j],
                    'sparsity': sparsity_scores[j],
                    'diversity': diversity_scores[j],
                    'grammar': grammar_scores[j],
                    'data_manifold_closeness_per_label_averaged': data_manifold_closeness_scores_per_label[j],
                    'data_manifold_closeness_for_topic': data_manifold_closeness_scores_per_topic[j]
                })
            # Append the list of counterfactuals and metrics to the main list
            counterfactuals_evaluation.append({
                'original_sentence': original_sentence_dict["sentences"][i],
                'source_label': original_sentence_dict["source_labels"][i],
                'target_label': list(original_sentence_dict["target_labels"][i]),
                'counterfactuals': iner_dict_list
            })

        if save_path is not None:
            self.save_eval_to_json(counterfactuals_evaluation, save_path)

        return counterfactuals_evaluation

    def save_eval_to_json(self, eval_dict: Dict, save_path: pathlib.Path):
        """
        Save the evaluation dictionary to a JSON file.
        :param eval_dict: dictionary containing the evaluation results from self.evaluate()
        """
        # Create the main dictionary to store the information
        data = {
            'counterfactuals_evaluation': eval_dict
        }

        # Write the data dictionary to the JSON file
        with save_path.open(mode="w") as json_file:
            json.dump(obj=data, fp=json_file, indent=4, sort_keys=False)

        return data

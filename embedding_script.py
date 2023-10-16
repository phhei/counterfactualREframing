"""
Script to generate sentence embeddings with different strategies.
Either the whole sentence is used, only the relevant tokens are used (keep other),
the relevant tokens are emphasized or foreign topics are removed (remove other).
"""
from typing import List

import pandas as pd
import torch
import yaml
from sentence_transformers import SentenceTransformer
import pickle
import os

from tqdm import tqdm

from Utils import get_label_saved_name
from dataset_loader import DatasetLoader


def get_sentences_by_strategy(df: pd.DataFrame,
                              strategy: str,
                              other_labels_options: List[str]):
    """
    Gets whole sentences per aspect with a given strategy.
    :param df: dataframe with sentences and labels
    :param strategy: strategy to get the sentences
    :param other_labels_options: string of 'other' label
    :return:  aspect_to_sentences dict with sentences per aspect where the sentences are filtered by the strategy
    """

    def get_whole_sentence(tokens):
        """
        Gets the whole sentence as a list of tokens.
        :param tokens:
        :return: original sentence as list of tokens
        """
        return tokens.tolist()

    def get_sentence_foreign_topics_removed(tokens, labels, main_topic):
        """
        Gets the whole sentence as a list of tokens, but removes foreign topics. (keep "other" token)
        :param tokens: all tokens of the sentence
        :param labels: all labels of the sentence
        :param main_topic: main topic of the sentence to look out for
        :return: filtered sentence as list of tokens
        """
        keep_tokens = []
        for token, label in zip(tokens, labels):
            if label == main_topic or label in other_labels_options:
                keep_tokens.append(token)
        return keep_tokens

    def get_sentence_only_selected_aspect(tokens, labels, selected_aspect):
        """
        Gets the only main token in sentence as a list of tokens. (remove "other" token)
        :param tokens: all tokens of the sentence
        :param labels: all labels of the sentence
        :param selected_aspect: main aspect of the sentence to look out for
        :return: filtered sentence as list of tokens
        """
        keep_tokens = []
        for token, label in zip(tokens, labels):
            if label == selected_aspect:
                keep_tokens.append(token)
        return keep_tokens

    def get_sentence_repeated_main_topic(tokens, labels, main_topic):
        """
        Gets the whole sentence as a list of tokens, but repeats the main topic.
        :param tokens: all tokens of the sentence
        :param labels: all labels of the sentence
        :param main_topic: main topic of the sentence to look out for
        :return: extended sentence as list of tokens
        """
        keep_tokens = tokens.tolist()
        keep_tokens.append(".")
        for token, label in zip(tokens, labels):
            if label == main_topic:
                keep_tokens.append(token)
        return keep_tokens

    # iterate over sentences
    aspect_to_sentences = {}
    for i, row in df.iterrows():
        unique_labels = set(row["labels"])
        for label in unique_labels:
            if label not in other_labels_options:
                # Get correct sentence representation
                if strategy == "whole_sentences":
                    sentence = get_whole_sentence(row["tokens"])
                elif strategy == "remove_foreign_topics":
                    sentence = get_sentence_foreign_topics_removed(row["tokens"], row["labels"], label)
                elif strategy == "only_relevant_tokens":
                    sentence = get_sentence_only_selected_aspect(row["tokens"], row["labels"], label)
                else:  # strategy == "emphasize_relevant_tokens":
                    sentence = get_sentence_repeated_main_topic(row["tokens"], row["labels"], label)
                # Save it to dict
                try:
                    aspect_to_sentences[label].append(" ".join(sentence))
                except KeyError:
                    aspect_to_sentences[label] = [" ".join(sentence)]
    return aspect_to_sentences


def main(global_config):
    sbert_model_name = global_config.get("s_bert_model_name", "all-MiniLM-L6-v2")
    model = SentenceTransformer(
        model_name_or_path=sbert_model_name,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    topic_list = ["mj", "mw", "ne"]
    strategies_list = ["emphasize_relevant_tokens", "remove_foreign_topics", "whole_sentences", "only_relevant_tokens",
                       "whole_topic"]

    for topic in tqdm(topic_list, desc="Topics"):
        for strategy in tqdm(strategies_list, desc=f"Strategy in {topic}"):
            dataset_loader = DatasetLoader()
            data_path = f"datasets/{topic}_conll_{'gold' if global_config.get('load_ground_truth', False) else 'silver'}_chunk.txt"
            ds, labels_set = dataset_loader.load_dataset(data_path)
            # o_label_idx = label_idx_mapping["O"]  # Save to not store as topic.
            df = ds.to_pandas()

            # Get sentences by strategy and aspect
            if strategy != "whole_topic":
                directory_path = f'sbert_embeddings/{sbert_model_name}/{strategy}/{topic}'
                os.makedirs(directory_path, exist_ok=True)
                aspect_to_sentences_dict = get_sentences_by_strategy(df, strategy, global_config['other_labels'])
                for aspect, sentences in aspect_to_sentences_dict.items():
                    aspect_embeddings = model.encode(sentences, convert_to_numpy=True)
                    # Store sentences & embeddings on disc
                    aspect = get_label_saved_name(aspect)
                    with open(f'{directory_path}/{aspect}.pkl', "wb+") as fOut:
                        pickle.dump({'embeddings': aspect_embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                directory_path = f"sbert_embeddings/{sbert_model_name}/{strategy}"
                os.makedirs(directory_path, exist_ok=True)
                sentences = df["tokens"].tolist()
                topic_embeddings = model.encode(sentences, convert_to_numpy=True)
                # Store sentences & embeddings on disc
                with open(f'{directory_path}/{topic}.pkl', "wb+") as fOut:
                    pickle.dump({'embeddings': topic_embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    with open('global_settings.yml', 'r') as file:
        global_config = yaml.safe_load(file)
    main(global_config)

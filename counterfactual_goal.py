import random


def get_sentence_to_target_label_dict(data, strategy, k, label_set, other_labels_list):
    """
    Function defines the target set for the counterfactual generation.
    Create a dict of sentences with source labels and target labels given a masking strategy and k.
    :param data: dict of sentences and labels
    :param strategy: masking strategy
    :param k: number of labels to remove/ exchange/ add (depending on the strategy)
    :param label_set: set of all possible labels
    :param other_labels_list: list of labels to ignore
    :return: dict of sentences with source labels and target labels
    """
    label_set = {label for label in label_set if label not in other_labels_list}  # ignore other labels
    sentence_labels_dict = {'sentences': [], 'source_labels': [], 'target_labels': []}
    for sentence, current_labels in zip(data["tokens"], data["labels"]):
        current_labels_set = set(
            [label for label in current_labels if label not in other_labels_list])  # ignore other labels
        target_label_set = set(current_labels_set)
        if strategy == "add_random_labels":
            # choose k random labels that are not already in the sentence
            random_labels = random.sample(label_set - current_labels_set, k)
            target_label_set.update(set(random_labels))
        elif strategy == "remove_random_labels":
            # check that there are at least k labels in the sentence
            if len(current_labels_set) <= k:
                continue  # ignore this sentence
            # choose k random labels that are in the sentence to remove
            random_labels = random.sample(current_labels_set, k)
            target_label_set.difference_update(set(random_labels))
        elif strategy == "exchange_random_labels":
            if len(current_labels_set) < k:
                continue
            # choose k random labels that are in the sentence
            random_labels = random.sample(current_labels_set, k)
            # choose k random labels that are not already in the sentence
            random_labels_to_add = random.sample(label_set - current_labels_set, k)
            # Remove from target labels and add new labels
            target_label_set.difference_update(set(random_labels))
            target_label_set.update(set(random_labels_to_add))
        else:
            raise ValueError(f"Invalid strategy: {strategy}")
        sentence_labels_dict['sentences'].append(sentence)
        sentence_labels_dict['source_labels'].append(current_labels)
        sentence_labels_dict['target_labels'].append(target_label_set)
    return sentence_labels_dict

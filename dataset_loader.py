# Datasets need to be downloaded from https://github.com/Leibniz-HBI/argument-aspect-corpus-v1
from typing import Set, Tuple

from datasets import Dataset


class DatasetLoader:
    def __init__(self,
                 map_labels_to_ids=False):
        """
        :param map_labels_to_ids: if True, labels are mapped to ids
        """
        self.map_labels_to_ids = map_labels_to_ids
        pass

    def load_dataset(self, data_path) -> Tuple[Dataset, Set]:
        def read_txt_file(folder_path, label_set: Set):
            filepath = f"{folder_path}"
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            dict_entries = []
            sentence = []
            sentence_counter = 0
            for line in lines:
                line = line.strip()
                if line:  # non-empty line
                    sentence.append(line.split("\t"))
                elif sentence:  # empty line and not the first line of the file
                    dict_entries.append(
                        {
                            "tokens": [triple[1] for triple in sentence],
                            "labels": [triple[2] for triple in sentence],
                        }
                    )
                    label_set.update([triple[2] for triple in sentence])
                    sentence = []
                    sentence_counter += 1
            return dict_entries, label_set

        def convert_labels_to_ids():
            label_idx_mapping = {}
            idx_label_mapping = {}
            for idx, label in enumerate(label_set):
                label_idx_mapping[label] = idx
                idx_label_mapping[idx] = label

            for entry in dataset_list:
                entry["labels"] = [label_idx_mapping[label] for label in entry["labels"]]  # turns labels to ints
            return label_idx_mapping, idx_label_mapping

        label_set = set()
        # load data from txt file
        dataset_list, label_set = read_txt_file(data_path, label_set)
        ds = Dataset.from_list(dataset_list)

        if self.map_labels_to_ids:  # Whether to leave labels as strings or convert to ids
            label_idx_mapping, idx_label_mapping = convert_labels_to_ids()
            return ds, label_idx_mapping, idx_label_mapping

        return ds, label_set

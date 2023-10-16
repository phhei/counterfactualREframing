from collections import defaultdict
from json import load
from pathlib import Path
from random import sample
from typing import List, Dict

from tqdm import tqdm

class SQLInsertObject:
    def __init__(self, topic: str, topic_long: str, original_text: str, counterfactual: str, experiment_type: str,
                 original_frames: str, target_frames: str):
        self.topic = topic
        self.topic_long = topic_long
        self.original_text = original_text.replace("'", "\\'")
        self.counterfactual = counterfactual.replace("'", "\\'")
        self.experiment_type = experiment_type
        self.original_frames = original_frames
        self.target_frames = target_frames

    def __hash__(self):
        return hash((self.topic, self.original_text, self.counterfactual))

    def __eq__(self, other):
        return (isinstance(other, SQLInsertObject) and self.topic == other.topic
                and self.original_text == other.original_text and self.counterfactual == other.counterfactual)

    def merge(self, other):
        if self == other:
            self.experiment_type += f"//{other.experiment_type}"
            self.target_frames += f"//{other.target_frames}"
            return True
        return False


if __name__ == "__main__":
    strategies = [
        # ("ne/remove_random_labels_1_by_t5-small", False),
        ("ne/remove_random_labels_1_by_t5-small", True),
        ("ne/exchange_random_labels_1_by_t5-small", False),
        ("ne/exchange_random_labels_1_by_t5-small", True),
        ("ne/remove_random_labels_1_by_t5-large", False),
        ("ne/exchange_random_labels_1_by_t5-large", False),
        ("ne/framed_decoding0.1/exchange_random_labels_1_by_t5-small", True),
        ("ne/framed_decoding0.1/exchange_random_labels_1_by_t5-large", False),
        # ("ne/framed_decoding0.2/remove_random_labels_1_by_t5-small", False),
        ("ne/framed_decoding0.2/remove_random_labels_1_by_t5-small", True),
        ("ne/framed_decoding0.2/exchange_random_labels_1_by_t5-small", False),
        ("ne/framed_decoding0.2/exchange_random_labels_1_by_t5-small", True),
        # ("ne/framed_decoding0.2/exchange_random_labels_2_by_t5-small", False),
        ("ne/framed_decoding0.2/exchange_random_labels_2_by_t5-small", True),
        ("ne/framed_decoding0.2/remove_random_labels_1_by_t5-large", False),
        ("ne/framed_decoding0.2/exchange_random_labels_1_by_t5-large", False),
        ("ne/framed_decoding0.2/exchange_random_labels_2_by_t5-large", False)
    ]
    samples_per_strategy = 50

    sample_list = sample(range(1000), samples_per_strategy*3)
    insert_dict: Dict[SQLInsertObject, List[SQLInsertObject]] = defaultdict(list)

    for strategy, consider_only_first_10_cfs in tqdm(strategies):
        added_samples = 0
        with Path(f"../../results/_gpu_results/{strategy}.json").open(mode="r") as f:
            data: List = load(f)["counterfactuals_evaluation"]

        for index in sample_list:
            try:
                data_point = data[index]
            except IndexError:
                continue

            counterfactuals_list: List = data_point["counterfactuals"][:10]\
                if consider_only_first_10_cfs else data_point["counterfactuals"]

            if len(data_point["original_sentence"]) >= 325:
                continue
            if any(map(lambda c: c["cf_string"] == "RUNTIME-ERROR", counterfactuals_list)):
                continue

            counterfactuals_list.sort(
                key=lambda c: 4*c["validity"]+2*c["grammar"]+c["proximity"]+c["data_manifold_closeness_for_topic"]+.5*c['data_manifold_closeness_per_label_averaged'],
                reverse=True
            )
            #print(data["counterfactuals"].index(counterfactuals_list[0]))

            if len(counterfactuals_list[0]["cf_string"]) >= 325:
                continue

            sql_insert_object = SQLInsertObject(
                topic=strategy[:2],
                topic_long="Marijuana" if strategy[:2] == "mj" else ("Minimum wage" if strategy[:2] == "mw" else "Nuclear energy"),
                original_text=data_point["original_sentence"],
                counterfactual=counterfactuals_list[0]["cf_string"],
                experiment_type=f"{strategy}_TOP{len(counterfactuals_list)}",
                original_frames="-".join(set(data_point["source_label"])),
                target_frames="-".join(data_point["target_label"])
            )

            insert_dict[sql_insert_object].append(
                sql_insert_object
            )
            added_samples += 1
            if added_samples >= samples_per_strategy:
                break

    insert_values: List[SQLInsertObject] = []
    for val in insert_dict.values():
        if len(val) == 1:
            insert_values.append(val[0])
        else:
            print("Merging {} objects".format(len(val)))
            _val = val.pop(0)
            for v in val:
                _val.merge(v)
            insert_values.append(_val)

    insert_text = ("INSERT INTO ReframeCounterfactualData"
                   "(topic, topic_long, original_text, counterfactual, experiment_type, "
                   "original_frames, target_frames) VALUES {};").format(
        ",\n".join(map(lambda v: f"('{v.topic}', '{v.topic_long}', '{v.original_text}', '{v.counterfactual}', "
                                 f"'{v.experiment_type}', '{v.original_frames}', '{v.target_frames}')", insert_values))
    )
    Path(f"INSERT-{'+'.join(map(lambda strat: '{}-{}'.format(strat[0].replace('/', '-'), strat[1]), strategies[:3]))}{'...' if len(strategies) > 3 else ''}.sql").write_text(data=insert_text, encoding="utf-8")


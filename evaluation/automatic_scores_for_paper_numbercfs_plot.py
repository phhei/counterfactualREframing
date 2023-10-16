from collections import defaultdict
from itertools import chain
from json import load
from typing import List

from tqdm import tqdm

from evaluation.automatic_scores_for_paper import get_stats_for_single_sample

from loguru import logger
from pathlib import Path

from matplotlib import pyplot

if __name__ == "__main__":
    path_to_explore: List[Path] = [
        Path("../results/_gpu_results/mj/exchange_random_labels_1_by_t5-large.json"),
        Path("../results/_gpu_results/mj/framed_decoding0.2/exchange_random_labels_1_by_t5-large.json")
    ]

    logger.info("OK, let's log - crawling through \"{}\"", " and ".join(map(lambda p: p.name, path_to_explore)))

    series = dict()
    for json_file in tqdm(iterable=path_to_explore, desc="Processing JSON files", unit="file"):
        with json_file.open(mode="r", encoding="utf-8") as f:
            logger.debug("Loading data from \"{}\" ({})", json_file.name, f.encoding)
            data: List = load(f)["counterfactuals_evaluation"]

        def extract_series() -> List:
            data_having_counterfactuals = list(filter(lambda cl: all(map(lambda c: c["cf_string"] != "RUNTIME-ERROR",
                                                                         cl["counterfactuals"])), data))

            ret = defaultdict(lambda: defaultdict(list))
            for sample in data_having_counterfactuals:
                for cut in range(1, len(sample["counterfactuals"])+1):
                    _successful_reframed, _validity, _all = get_stats_for_single_sample(
                        keys_for_sorting={"validity": 4 / 8.5, "grammar": 2 / 8.5, "proximity": 1 / 8.5,
                                          "data_manifold_closeness_for_topic": 1 / 8.5,
                                          "data_manifold_closeness_per_label_averaged": .5 / 8.5},
                        counterfactuals_list=sample["counterfactuals"],
                        top_k=1,
                        f_cut=cut,
                        keys_for_calculating=[
                            {"successful_reframed": 1},
                            {"validity": 1},
                            {"validity": 1 / 5, "grammar": 1 / 5, "proximity": 1 / 5,
                             "data_manifold_closeness_for_topic": 1 / 5,
                             "data_manifold_closeness_per_label_averaged": 1 / 5}
                        ],
                        target_frames=set(sample["target_label"])
                    )
                    ret["successful reframed"][cut].append(_successful_reframed)
                    # ret["validity"][cut].append(_validity)
                    ret["average score"][cut].append(_all)

            return {metric: [(cut, sum(single_values)/len(single_values))
                             for cut, single_values in values.items()] for metric, values in ret.items()}

        series[f"{json_file.stem[json_file.stem.rindex('_')+1:]} (frame-dec: {0 if json_file.parent.name in ['mj', 'mw', 'ne'] else json_file.parent.name[json_file.parent.name.index('0'):]})"] = extract_series()

    for model_i, (model, values) in enumerate(series.items()):
        logger.debug("Plotting {} - metrics: {}", model, values.keys())
        for metric, cut_value_list in values.items():
            logger.trace("Plotting {} - {} - cut_value_list: {}", model, metric, cut_value_list)
            ax = pyplot.plot(
                [cut for cut, _ in cut_value_list],
                [value*100 for _, value in cut_value_list],
                "-" if model_i == 0 else ("--" if model_i == 1 else ("-." if model_i == 2 else ":")),
                label=f"{model} - {metric}",
                scaley=False,
            )
            pyplot.legend()
            pyplot.ylim(0, 100)
            pyplot.xlabel("number of generated candidates")
            pyplot.grid(color='black', alpha=0.2, linestyle='-', linewidth=1, axis="y")
            pyplot.yticks(range(0, 101, 10), [f"{i}%" for i in range(0, 101, 10)])
    pyplot.show()

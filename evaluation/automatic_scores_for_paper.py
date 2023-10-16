from collections import defaultdict
from json import load
from pathlib import Path
from typing import List, Dict, Optional, Union
from re import findall

import pandas
from tqdm import tqdm


def get_stats_for_single_sample(
        keys_for_sorting: Dict[str, float],
        counterfactuals_list: List[Dict],
        top_k: int,
        f_cut: Optional[int],
        keys_for_calculating: Optional[Union[Dict[str, float], List[Dict[str, float]]]] = None,
        **kwargs
) -> Union[float, List[float]]:
    keys_for_calculating = keys_for_calculating or keys_for_sorting
    counterfactuals_list = counterfactuals_list[:min(f_cut or len(counterfactuals_list), len(counterfactuals_list))]
    counterfactuals_list.sort(
        key=lambda c: sum(map(lambda kv: kv[1] * c[kv[0]], keys_for_sorting.items())),
        reverse=True
    )
    if isinstance(keys_for_calculating, Dict):
        keys_for_calculating = [keys_for_calculating]
        return_first_element = True
    else:
        return_first_element = False

    ret_floats = list()
    for keys_for_calculating_single in keys_for_calculating:
        try:
            ret_floats.append(sum(
                map(lambda c: sum(map(lambda kv: kv[1] * c[kv[0]], keys_for_calculating_single.items())),
                    counterfactuals_list[:top_k])
            ) / top_k)
        except KeyError:
            f_ret = 0
            if "successful_removed" in keys_for_calculating_single:
                f_ret += keys_for_calculating_single["successful_removed"] * sum(
                    map(lambda c: int(all(map(lambda pl: pl not in kwargs.get("negative_frames", []),
                                              findall(r"[A-Z]+", c["pred_labels_set"])))),
                        counterfactuals_list[:top_k])
                ) / top_k
            if "successful_added" in keys_for_calculating_single:
                f_ret += keys_for_calculating_single["successful_added"] * sum(
                    map(lambda c: int(all(map(lambda pf: pf in findall(r"[A-Z]+", c["pred_labels_set"]),
                                              kwargs.get("positive_frames", [])))),
                        counterfactuals_list[:top_k])
                ) / top_k
            if "successful_reframed" in keys_for_calculating_single:
                f_ret += keys_for_calculating_single["successful_reframed"] * sum(
                    map(lambda c: int(set(findall(r"[A-Z]{2,}", c["pred_labels_set"])) ==
                                      kwargs.get("target_frames", set())),
                        counterfactuals_list[:top_k])
                ) / top_k
            ret_floats.append(f_ret)

    return ret_floats[0] if return_first_element else ret_floats


if __name__ == "__main__":
    base_path = Path("../results/_gpu_results")

    ret = defaultdict(dict)
    json_list = list(base_path.rglob(pattern="*.json"))
    for json_file in tqdm(iterable=json_list, desc="Processing JSON files", unit="file"):
        with json_file.open(mode="r") as f:
            data: List = load(f)["counterfactuals_evaluation"]

        def stats() -> Dict:
            data_having_counterfactuals = list(filter(lambda cl: all(map(lambda c: c["cf_string"] != "RUNTIME-ERROR",
                                                             cl["counterfactuals"])), data))
            stats_ret = dict()
            for cut in [1, 2, 5, 10, None]:
                cut_str = f"CUT{cut:02d}" if cut is not None else "ALL"
                stats_ret.update({
                    f"_succreframed_percentage_TOPval_{cut_str}": sum(
                        map(lambda sample: get_stats_for_single_sample(
                            keys_for_sorting={"validity": 1},
                            counterfactuals_list=sample["counterfactuals"],
                            top_k=1,
                            f_cut=cut,
                            keys_for_calculating={"successful_reframed": 1},
                            target_frames=set(sample["target_label"])
                        ), data_having_counterfactuals)) / len(data_having_counterfactuals),
                    f"_succreframed_percentage_TOPcf_{cut_str}": sum(
                        map(lambda sample: get_stats_for_single_sample(
                            keys_for_sorting={"validity": 4 / 8.5, "grammar": 2 / 8.5, "proximity": 1 / 8.5,
                                              "data_manifold_closeness_for_topic": 1 / 8.5,
                                              "data_manifold_closeness_per_label_averaged": .5 / 8.5},
                            counterfactuals_list=sample["counterfactuals"],
                            top_k=1,
                            f_cut=cut,
                            keys_for_calculating={"successful_reframed": 1},
                            target_frames=set(sample["target_label"])
                        ), data_having_counterfactuals)) / len(data_having_counterfactuals),
                    f"_frames_succdeleted_percentage_TOPval_{cut_str}": sum(
                        map(lambda sample: get_stats_for_single_sample(
                            keys_for_sorting={"validity": 1},
                            counterfactuals_list=sample["counterfactuals"],
                            top_k=1,
                            f_cut=cut,
                            keys_for_calculating={"successful_removed": 1},
                            negative_frames=set(sample["source_label"]) - set(sample["target_label"]) - {"O"}
                        ), data_having_counterfactuals)) / len(data_having_counterfactuals),
                    f"_frames_succadded_percentage_TOPval_{cut_str}": sum(
                        map(lambda sample: get_stats_for_single_sample(
                            keys_for_sorting={"validity": 1},
                            counterfactuals_list=sample["counterfactuals"],
                            top_k=1,
                            f_cut=cut,
                            keys_for_calculating={"successful_added": 1},
                            positive_frames=set(sample["target_label"]) - set(sample["source_label"]) - {"O"}
                        ), data_having_counterfactuals)) / len(data_having_counterfactuals),
                    f"_validity_TOPval_{cut_str}": sum(map(lambda sample:
                        get_stats_for_single_sample(keys_for_sorting={"validity": 1},
                                                    counterfactuals_list=sample["counterfactuals"], top_k=1,
                                                    f_cut=cut), data_having_counterfactuals)) / len(data_having_counterfactuals),
                    f"validity_TOP2val_{cut_str}": sum(map(lambda sample:
                        get_stats_for_single_sample(keys_for_sorting={"validity": 1},
                                                    counterfactuals_list=sample["counterfactuals"],
                                                    top_k=min(2, cut or 50),
                                                    f_cut=cut), data_having_counterfactuals)) / len(data_having_counterfactuals),
                    f"validity_TOPhalfval_{cut_str}": sum(map(lambda sample:
                        get_stats_for_single_sample(keys_for_sorting={"validity": 1},
                                                    counterfactuals_list=sample["counterfactuals"],
                                                    top_k=max(1, (cut or len(sample["counterfactuals"]))//2),
                                                    f_cut=cut), data_having_counterfactuals)) / len(data_having_counterfactuals),
                    f"proximity_TOPprox_{cut_str}": sum(map(lambda sample:
                        get_stats_for_single_sample(keys_for_sorting={"proximity": 1},
                                                    counterfactuals_list=sample["counterfactuals"], top_k=1,
                                                    f_cut=cut), data_having_counterfactuals)) / len(data_having_counterfactuals),
                    f"sparsity_TOP1spars_{cut_str}": sum(map(lambda sample:
                        get_stats_for_single_sample(keys_for_sorting={"sparsity": 1},
                                                    counterfactuals_list=sample["counterfactuals"], top_k=1,
                                                    f_cut=cut), data_having_counterfactuals)) / len(data_having_counterfactuals),
                    f"diversity_TOPdiv_{cut_str}": sum(map(lambda sample:
                        get_stats_for_single_sample(keys_for_sorting={"diversity": 1},
                                                    counterfactuals_list=sample["counterfactuals"], top_k=1,
                                                    f_cut=cut), data_having_counterfactuals)) / len(data_having_counterfactuals),
                    f"grammar_TOPgram_{cut_str}": sum(map(lambda sample:
                        get_stats_for_single_sample(keys_for_sorting={"grammar": 1},
                                                    counterfactuals_list=sample["counterfactuals"], top_k=1,
                                                    f_cut=cut), data_having_counterfactuals)) / len(data_having_counterfactuals),
                    f"data_manifold_closeness_per_label_averaged_TOPdl_{cut_str}": sum(map(lambda sample:
                        get_stats_for_single_sample(keys_for_sorting={"data_manifold_closeness_per_label_averaged": 1},
                                                    counterfactuals_list=sample["counterfactuals"], top_k=1,
                                                    f_cut=cut), data_having_counterfactuals)) / len(data_having_counterfactuals),
                    f"data_manifold_closeness_for_topic_TOPdt_{cut_str}": sum(map(lambda sample:
                        get_stats_for_single_sample(keys_for_sorting={"data_manifold_closeness_for_topic": 1},
                                                    counterfactuals_list=sample["counterfactuals"], top_k=1,
                                                    f_cut=cut), data_having_counterfactuals)) / len(data_having_counterfactuals),
                    f"_avg_score_TOPcf_{cut_str}": sum(map(lambda sample: get_stats_for_single_sample(
                        keys_for_sorting={"validity": 4 / 8.5, "grammar": 2 / 8.5, "proximity": 1 / 8.5,
                                          "data_manifold_closeness_for_topic": 1 / 8.5,
                                          "data_manifold_closeness_per_label_averaged": .5 / 8.5},
                        counterfactuals_list=sample[ "counterfactuals"],
                        top_k=1,
                        f_cut=cut,
                        keys_for_calculating={"validity": 1 / 5, "grammar": 1 / 5, "proximity": 1 / 5,
                                              "data_manifold_closeness_for_topic": 1 / 5,
                                              "data_manifold_closeness_per_label_averaged": 1 / 5}
                    ), data_having_counterfactuals)) / len(data_having_counterfactuals),
                    f"avg_score_TOP2cf_{cut_str}": sum(map(lambda sample: get_stats_for_single_sample(
                        keys_for_sorting={"validity": 4 / 8.5, "grammar": 2 / 8.5, "proximity": 1 / 8.5,
                                          "data_manifold_closeness_for_topic": 1 / 8.5,
                                          "data_manifold_closeness_per_label_averaged": .5 / 8.5},
                        counterfactuals_list=sample["counterfactuals"],
                        top_k=min(2, cut or 50),
                        f_cut=cut,
                        keys_for_calculating={"validity": 1 / 5, "grammar": 1 / 5, "proximity": 1 / 5,
                                              "data_manifold_closeness_for_topic": 1 / 5,
                                              "data_manifold_closeness_per_label_averaged": 1 / 5}
                    ), data_having_counterfactuals)) / len(data_having_counterfactuals),
                    f"validity_TOPcf_{cut_str}": sum(map(lambda sample: get_stats_for_single_sample(
                        keys_for_sorting={"validity": 4, "grammar": 2, "proximity": 1,
                                          "data_manifold_closeness_for_topic": 1,
                                          "data_manifold_closeness_per_label_averaged": .5},
                        counterfactuals_list=sample["counterfactuals"],
                        top_k=1,
                        f_cut=cut,
                        keys_for_calculating={"validity": 1}
                    ), data_having_counterfactuals)) / len(data_having_counterfactuals),
                    f"grammar_TOPcf_{cut_str}": sum(map(lambda sample: get_stats_for_single_sample(
                        keys_for_sorting={"validity": 4, "grammar": 2, "proximity": 1,
                                          "data_manifold_closeness_for_topic": 1,
                                          "data_manifold_closeness_per_label_averaged": .5},
                        counterfactuals_list=sample["counterfactuals"],
                        top_k=1,
                        f_cut=cut,
                        keys_for_calculating={"grammar": 1}
                    ), data_having_counterfactuals)) / len(data_having_counterfactuals),
                })

            return stats_ret

        if json_file.parent.name.startswith("framed_decoding"):
            ret[json_file.parent.parent.name][f"{json_file.stem}_dec{json_file.parent.name[len('framed_decoding'):]}"] \
                = stats()
        else:
            ret[json_file.parent.name][json_file.stem] = stats()

    for topic, results in ret.items():
        df = pandas.DataFrame.from_dict(data=results, orient="index")
        df = df.reindex(columns=sorted(df.columns))
        df.to_csv(base_path.joinpath(f"results_{topic}.csv"), float_format="%.3f", encoding="utf-8")
        # heatmap = heatmap_fn(
        #    data=df, annot=True, fmt=".3f", vmin=0, vmax=1, cbar=False
        #)
        # heatmap.set_title(f"Topic: {topic}")
        # heatmap.get_figure().savefig(f"heatmap_{topic}.png")
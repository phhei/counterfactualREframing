from collections import defaultdict, Counter
from typing import Tuple

import pandas
import numpy

from loguru import logger
from pathlib import Path
from pprint import pformat
from json import dump as dump_json, JSONEncoder
from krippendorff import alpha as krippendorff_alpha

from tqdm import tqdm

annotation_file = Path("annotations_extended.csv")


class JSONEncoderWithNumpy(JSONEncoder):
    def default(self, o):
        if isinstance(o, numpy.int64):
            return int(o)
        return super().default(o)


if __name__ == "__main__":
    logger.debug("Loading annotations from \"{}\"", annotation_file.name)

    annotations = pandas.read_csv(annotation_file, sep="|", encoding="utf-8", index_col=False)
    logger.info("Loaded {} annotations", len(annotations))

    ret = defaultdict(dict)

    logger.debug("OK, let's calculate the aggregated stats first")
    sample_groups = annotations.groupby(by=["topic", "original_text", "counterfactual"], axis="rows", as_index=True)
    logger.debug("Found {} different samples", len(sample_groups))
    interesting_columns = [
        ("Waste", ("a_frames", "WASTE"), "nominal"),
        ("Accidents/security", ("a_frames", "ACCIDENTS/SECURITY"), "nominal"),
        ("Health effects", ("a_frames", "HEALTH"), "nominal"),
        ("Environmental impact", ("a_frames", "ENVIRONMENTAL"), "nominal"),
        ("Costs", ("a_frames", "COSTS"), "nominal"),
        ("Weapons", ("a_frames", "WEAPONS"), "nominal"),
        ("Reliability/efficiency", ("a_frames", "RELIABILITY"), "nominal"),
        ("Technological innovation", ("a_frames", "TECHNOLOGICAL"), "nominal"),
        ("Renewables", ("a_frames", "RENEWABLES"), "nominal"),
        ("Fossil fuels", ("a_frames", "FOSSIL"), "nominal"),
        ("Energy policy", ("a_frames", "ENERGY"), "nominal"),
        ("Public debate", ("a_frames", "PUBLIC"), "nominal"),
        ("fluency", ("a_fluency", None), "ordinal"),
        ("meaning", ("a_meaning", None), "ordinal"),
        ("quality", ("a_meaning", (0, 10)), "nominal"),
        ("meaning-preservation", ("a_meaning", (1, 10)), "nominal"),
    ]
    for name, (col, matching_value), level_of_measurement in interesting_columns:
        reliable_data = defaultdict(list)
        for sample, annotated_sample in sample_groups:
            logger.trace("Processing sample \"{}\" for calculating inter-annotator-agreement", sample)
            for _, annotation in annotated_sample.iterrows():
                if matching_value is None:
                    reliable_data[annotation["annotator_ID"]].append(annotation[col])
                elif isinstance(matching_value, Tuple):
                    reliable_data[annotation["annotator_ID"]].append(int(annotation[col] in matching_value))
                else:
                    reliable_data[annotation["annotator_ID"]].append(int(matching_value in annotation[col]))
        logger.debug("Calculating krippendorff alpha having {} annotators for \"{}\"", list(reliable_data.keys()), name)
        ret["_agreement_krippendorff_alpha"][name] = krippendorff_alpha(
            reliability_data=list(reliable_data.values()),
            level_of_measurement=level_of_measurement
        )
        logger.info("Calculated krippendorff alpha for \"{}\" is {}",
                    name, str(round(ret["_agreement_krippendorff_alpha"][name], 3)))

    multi_samples = annotations[annotations["experiment_type"].str.contains("//")]
    logger.debug("Found {} multi-samples", len(multi_samples))

    annotations.drop(index=multi_samples.index, inplace=True)
    new_lines = []
    for index, row in multi_samples.iterrows():
        for experiment_type, target_frame in zip(row.pop("experiment_type").split("//"), row.pop("target_frames").split("//")):
            row["experiment_type"] = experiment_type
            row["target_frames"] = target_frame
            new_lines.append(row.copy())

    annotations: pandas.DataFrame = pandas.concat([annotations, pandas.DataFrame(new_lines)],
                                                  axis="rows", ignore_index=True)
    logger.success("Annotations now contains {} entries - all together", len(annotations))

    experiment_type_groups = annotations.groupby(by="experiment_type", axis="rows", as_index=True)
    logger.debug("Found {} different experiment types: {}",
                 len(experiment_type_groups), experiment_type_groups.indices.keys())

    for experiment_type, experiment_group in tqdm(iterable=experiment_type_groups,
                                                  total=len(experiment_type_groups),
                                                  desc="Processing experiment types",
                                                  unit="experiment"):
        logger.trace("Processing experiment type \"{}\"", experiment_type)
        logger.debug("Found {} annotations for experiment type \"{}\"", len(experiment_group), experiment_type)

        for axis in ["timeInS", "a_fluency", "a_meaning"]:
            ret[experiment_type][axis] = {
                "count": experiment_group[axis].count(),
                "mean": experiment_group[axis].mean(),
                "std": experiment_group[axis].std(),
                "min": experiment_group[axis].min(),
                "max": experiment_group[axis].max()
            }
            if axis != "timeInS":
                ret[experiment_type][axis]["bins"] = experiment_group[axis].value_counts().to_dict()

        frame_stats = defaultdict(lambda: defaultdict(list))

        annotator_groups = experiment_group.groupby(by=["topic", "original_text", "counterfactual"],
                                                    axis="rows", as_index=True)
        logger.debug("Found {} different samples", len(annotator_groups))
        for index, sample in annotator_groups:
            logger.trace("Processing sample \"{}\" for frame evaluation", index)
            logger.trace("Found {} annotations for sample \"{}\"", len(sample), index)

            original_frames = set(sample["original_frames"].iloc[0].split("-")).difference(["", "O"])
            target_frames = set(sample["target_frames"].iloc[0].split("-")).difference(["", "O"])
            annotated_frames=sample["a_frames"].map("-{}-".format).sum()
            annotated_frames_counted = Counter(annotated_frames.split("-"))
            logger.trace("Remove {} blanks. Mostly annotated frames: {}",
                         annotated_frames_counted.pop("", 0), annotated_frames_counted.most_common(n=1))

            for min_count in [1, 2, 3]:
                annotated_frames_actual = {frame for frame, counter in annotated_frames_counted.items()
                                           if counter >= min_count}
                frame_stats[f"min_{min_count}x_voted"]["number_annotated_frames"].append(len(annotated_frames_actual))
                frame_stats[f"min_{min_count}x_voted"]["number_successful_removed_frames"].append(
                    len(original_frames.difference(target_frames)) -
                    len(original_frames.difference(target_frames).intersection(annotated_frames_actual))
                )
                frame_stats[f"min_{min_count}x_voted"]["number_removed_frames_should_stay"].append(
                    len(original_frames.union(target_frames).difference(annotated_frames_actual))
                )
                frame_stats[f"min_{min_count}x_voted"]["number_successful_added_frames"].append(
                    len(target_frames.difference(original_frames).intersection(annotated_frames_actual))
                )
                frame_stats[f"min_{min_count}x_voted"]["bool_successful_removed_frames"].append(
                    int(original_frames.difference(target_frames).isdisjoint(annotated_frames_actual))
                )
                frame_stats[f"min_{min_count}x_voted"]["bool_successful_added_frames"].append(
                    int(target_frames.difference(original_frames).issubset(annotated_frames_actual))
                )
                frame_stats[f"min_{min_count}x_voted"]["bool_successful_hit_frame_set"].append(
                    int(target_frames == annotated_frames_actual)
                )
                frame_stats[f"min_{min_count}x_voted"]["number_overlap_frames"].append(
                    len(target_frames.intersection(annotated_frames_actual))
                )

        logger.debug("Successfully collected {}->{} frame stats", len(frame_stats), sum(map(len, frame_stats.values())))
        frame_stats = {
            agreement: {stat: pandas.Series(values) for stat, values in stats.items()}
            for agreement, stats in frame_stats.items()
        }
        ret[experiment_type]["frame_stats"] = {
            agreement: {
                stat: {
                    "count": values.count(),
                    "mean": values.mean(),
                    "std": values.std(),
                    "min": values.min(),
                    "max": values.max(),
                    "bins": values.value_counts().to_dict()
                } for stat, values in stats.items()
            }
            for agreement, stats in frame_stats.items()
        }

        logger.debug("Finished processing experiment type \"{}\"", experiment_type)

    logger.success("Finished processing all experiment types, collected {}->{} stats",
                   len(ret), sum(map(len, ret.values())))

    ret_pprint = pformat(
        ret,
        indent=2,
        compact=False,
        sort_dicts=True
    )

    logger.success("Here the stats:\n{}", ret_pprint)

    with annotation_file.with_suffix(".json").open(mode="w", encoding="utf-8") as f:
        dump_json(obj=ret, fp=f, indent=4, sort_keys=True, cls=JSONEncoderWithNumpy)

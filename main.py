import ast
import os
import pathlib
from collections import defaultdict
from itertools import chain
from typing import Dict, Any, List

import click
import pandas as pd
import torch.cuda
import yaml
from loguru import logger
from sentence_transformers import SentenceTransformer

from counterfactual_goal import get_sentence_to_target_label_dict
from dataset_loader import DatasetLoader
from evaluation.evaluation_metrics import EvaluationMetrics
from flair_frame_classifier import FrameClassifier
from t5_editor import T5Editor


def load_config(config_name):
    full_path = f"experiment_configs/{config_name}.yaml"
    with open(full_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def get_save_path(mask_strategy, mask_k, config):
    folder_path = f"results{'_using_ground_truth_spans' if config.get('load_ground_truth', False) else ''}/{config['dataset_name']}/"
    if config.get("activate_framed_decoding", False):
        folder_path += f"framed_decoding{config.get('frame_decoding_strength', 0.1)}/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Replace the k in the masking strategy be real k value for the name
    file_name = f"{mask_strategy}_{mask_k}_by_{config['t5_model_name']}.csv"
    file_path = os.path.join(folder_path, file_name)
    return folder_path, file_name, file_path


def generate_counterfactuals_for_all_sentences(dict_of_sentences, editor, config) -> Dict[str, List]:
    if len(dict_of_sentences['sentences']) == 0:
        logger.warning("No sentences to generate counterfactuals for!")
        return {
            "original_sentence": [],
            "source_labels": [],
            "source_labels_set": [],
            "target_labels": [],
            "counterfactuals": []
        }

    ret = defaultdict(list)

    loop_list = list(zip(dict_of_sentences['sentences'],
                         dict_of_sentences['source_labels'],
                         dict_of_sentences['target_labels']))
    logger.info("Let's generate some counterfactuals! ({})", len(dict_of_sentences['sentences']))
    loop_list.sort(key=lambda x: x[2])
    generate_batch_size = config.get("generate_batch_size", 8)
    batched_loop_list = []
    batch = []
    for original_sentence, source_labels, target_labels in loop_list:
        if len(batch) == 0:
            logger.trace("Starting new batch with target labels: {}", target_labels)
            batch.append((original_sentence, source_labels, target_labels))
        elif len(batch) >= generate_batch_size or batch[0][2] != target_labels:
            logger.trace("Close a batch: {}", batch[0][2])
            batched_loop_list.append(([b[0] for b in batch], [b[1] for b in batch], batch[0][2]))
            batch = [(original_sentence, source_labels, target_labels)]
        else:
            batch.append((original_sentence, source_labels, target_labels))
    if len(batch) > 0:
        batched_loop_list.append(([b[0] for b in batch], [b[1] for b in batch], batch[0][2]))
    logger.info(
        "Gathered {} batches of size {}-{}-{}",
        len(batched_loop_list),
        min(len_batch := [len(b[0]) for b in batched_loop_list]),
        round(sum(len_batch) / len(batched_loop_list), 1),
        max(len_batch)
    )

    for original_sentences, source_labels, target_labels in batched_loop_list:
        logger.trace("Generating counterfactuals for sentences: {} ({}->{})",
                     original_sentences, source_labels, target_labels)
        ret["original_sentence"].extend([" ".join(s) for s in original_sentences])
        ret["source_labels"].extend(source_labels)
        ret["source_labels_set"].extend([set(labels) for labels in source_labels])
        ret["target_labels"].extend([target_labels]*len(original_sentences))
        ret["counterfactuals"].extend(editor.generate_counterfactuals(
            original_sentences=original_sentences,
            original_labels=source_labels,
            target_labels=target_labels,
            num_cfs=config['num_cfs_per_sentence'],
            topic=config['dataset_name_full'],
            other_class_labels=config['other_labels']
        ))

    logger.success("Done generating counterfactuals! ({})", sum(map(len, ret["counterfactuals"])))
    return ret


class Experiment:
    def __init__(self, init_global_config: Dict[str, Any]):
        """
        :param init_global_config: global config that has shared properties for all experiments
        """
        self.global_config = init_global_config

    def create_counterfactuals_table(self):
        dataset_loader = DatasetLoader()

        if "dataset_name" in self.global_config:
            topics = [self.global_config["dataset_name"]]
            load_local_config = False
        else:
            topics = self.global_config.get("topics", ["mj"])
            load_local_config = True

        logger.info("OK, processing {} topics ({})", len(topics), topics)

        for topic in topics:  # Iterate over topics
            if load_local_config:
                experiment_config = load_config(topic)
                # Extend the experiment config with the global config
                experiment_config.update(self.global_config)
            else:
                experiment_config = self.global_config

            # Load models
            editor = T5Editor(
                model_name=experiment_config['t5_model_name'],
                **experiment_config.get("generation_args", dict())
            )

            # Load data
            data, label_set = dataset_loader.load_dataset(
                f"datasets/{experiment_config['dataset_name']}_conll_{'gold' if experiment_config.get('load_ground_truth', False) else 'silver'}_chunk.txt"
            )

            if experiment_config.get("activate_framed_decoding", False):
                logger.debug("Prepare activating framed decoding, having {} labeled sentences ({} tokens)",
                             len(data), sum(map(len, data["tokens"])))
                token_class_list = [[(token, label) for token, label in
                                     zip(data[row_id]["tokens"], data[row_id]["labels"])]
                                    for row_id in range(len(data))]
                editor.activate_frame_decoding(
                    framed_tokens=list(chain.from_iterable(token_class_list)),
                    frame_decoding_strength=experiment_config.get("frame_decoding_strength", 0.1)
                )
                logger.success("Activated framed decoding with a strength of {}",
                               experiment_config.get("frame_decoding_strength", 0.1))

            if experiment_config.get("limit_data", None) is None:
                logger.trace("Data is not limited: {} labeled sentences", len(data))
            else:
                data = data.select(range(experiment_config["limit_data"]))
                logger.warning("Data is limited to {} labeled sentences", len(data))

            frame_modification_strategies = [
                (
                    f"{str_strategy[:(firstIndex := str_strategy.index('_'))]}_"
                    f"{str_strategy[(secondIndex := str_strategy.index('_', firstIndex + 1)) + 1:]}",
                    int(str_strategy[firstIndex + 1:secondIndex])
                )
                for str_strategy in experiment_config["frame_modification_strategies"] if isinstance(str_strategy, str)
            ]

            for mask_strategy, mask_k in frame_modification_strategies:
                _, _, file_path = get_save_path(mask_strategy, mask_k, experiment_config)
                force_recomputation = experiment_config.get("force_recomputation", False)

                if not force_recomputation and pathlib.Path(file_path).exists():
                    logger.info("[{}] Skipping counterfactual-creation for {}->{} because it already exists",
                                topic, mask_strategy, mask_k)
                else:
                    if force_recomputation:
                        logger.debug("[{}] (Re)computing {}->{}", topic, mask_strategy, mask_k)

                    # Get a dict of sentences with source labels and target labels
                    label_dict_sentences = get_sentence_to_target_label_dict(
                        data, mask_strategy, mask_k, label_set, experiment_config['other_labels']
                    )
                    logger.info("{} out of {} samples ({}%) suits to the strategy {} ({})",
                                len(label_dict_sentences['sentences']),
                                len(data),
                                round(len(100 * label_dict_sentences['sentences']) / len(data), 0),
                                mask_strategy, mask_k)

                    # Generate counterfactuals for one sentence at a time
                    data_dict = generate_counterfactuals_for_all_sentences(
                        label_dict_sentences, editor,
                        experiment_config
                    )

                    try:
                        for i in range(experiment_config["num_cfs_per_sentence"]):
                            col_name = f'cf_{i + 1}'
                            data_dict[col_name] = [cf[i] for cf in data_dict["counterfactuals"]]
                        data_dict.pop("counterfactuals")
                    except IndexError:
                        logger.warning("No counterfactuals were generated! (0/{})", len(data))

                    # Create the DataFrame
                    df = pd.DataFrame(data_dict)

                    # Save the DataFrame as a csv file
                    df.to_csv(file_path, index=False)

    def evaluate_counterfactuals_table(self):
        if "dataset_name" in self.global_config:
            topics = [self.global_config["dataset_name"]]
            load_local_config = False
        else:
            topics = self.global_config.get("topics", ["mj"])
            load_local_config = True

        logger.info("OK, evaluating {} topics ({})", len(topics), topics)

        for topic in topics:
            if load_local_config:
                config = load_config(topic)
                # Extend the experiment config with the global config
                config.update(self.global_config)
            else:
                config = self.global_config

            evaluation_metrics = EvaluationMetrics(config)

            posix_temp = pathlib.PosixPath
            try:
                classifier = FrameClassifier(model_path=f"trained_models/{config['dataset_name']}/best-model.pt")
            except NotImplementedError:
                logger.opt(exception=True).warning(
                    "Working on Windows, using WindowsPath instead of PosixPath (see "
                    "https://stackoverflow.com/questions/57286486/i-cant-load-my-model-because"
                    "-i-cant-put-a-posixpath/68796747#68796747)"
                )
                pathlib.PosixPath = pathlib.WindowsPath
                classifier = FrameClassifier(model_path=f"trained_models/{config['dataset_name']}/best-model.pt")
            finally:
                pathlib.PosixPath = posix_temp

            s_bert = SentenceTransformer(
                model_name_or_path=config['s_bert_model_name'],
                device="cuda" if torch.cuda.is_available() else "cpu"
            )

            frame_modification_strategies = [
                (
                    f"{str_strategy[:(firstIndex := str_strategy.index('_'))]}_"
                    f"{str_strategy[(secondIndex := str_strategy.index('_', firstIndex+1))+1:]}",
                    int(str_strategy[firstIndex+1:secondIndex])
                )
                for str_strategy in config["frame_modification_strategies"] if isinstance(str_strategy, str)
            ]

            for mask_strategy, mask_k in frame_modification_strategies:
                # Evaluate the saved CSV file
                folder_path, file_name, file_path = get_save_path(mask_strategy, mask_k, config)
                json_file = pathlib.Path(folder_path, file_name.replace(".csv", ".json"))
                force_recomputation = config.get("force_recomputation", False)

                if not pathlib.Path(file_path).exists():
                    logger.error("Please run the counterfactual creation first! {}->{}-{}",
                                 topic, mask_strategy, mask_k)
                    continue

                if not force_recomputation and json_file.exists():
                    logger.info("[{}] Skipping evaluation for {}->{} because it already exists",
                                topic, mask_strategy, mask_k)
                    continue

                if force_recomputation:
                    logger.debug("[{}] (Re)computing {}->{}", topic, mask_strategy, mask_k)

                df = pd.read_csv(file_path)

                source_labels_as_correct_dtype = [ast.literal_eval(labels) for labels in df['source_labels'].to_list()]
                target_labels_as_correct_dtype = [ast.literal_eval(labels) for labels in df['target_labels'].to_list()]
                # get original sentences dict from df with original sentences, source labels and target labels
                original_sentences_dict = {'sentences': df['original_sentence'].tolist(),
                                           'source_labels': source_labels_as_correct_dtype,
                                           'target_labels': target_labels_as_correct_dtype}
                # get counterfactuals from df
                counterfactuals = df.iloc[:, 4:].values.tolist()

                # Evaluate and save metrics
                # get save folder and change file name
                evaluation_metrics.evaluate(
                    original_sentences_dict,
                    counterfactuals,
                    classifier,
                    s_bert,
                    json_file
                )


@logger.catch
@click.command()
@click.argument("config",
                type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path))
def run(config: pathlib.Path):
    with pathlib.Path("global_settings.yml").open(mode='r') as global_config_file:
        run_experiment_config = yaml.safe_load(global_config_file)

    logger.info("Let's read the config file: {}", config.name)
    try:
        with config.open(mode='r') as config_file:
            run_experiment_config.update(yaml.safe_load(config_file))
    except IOError:
        logger.opt(exception=True).error("Could not read the config file: {} -- ignore it", config.absolute())

    run_experiment = Experiment(run_experiment_config)
    run_experiment.create_counterfactuals_table()
    run_experiment.evaluate_counterfactuals_table()


def test():
    with open('global_settings.yml', 'r') as file:
        global_config = yaml.safe_load(file)

    experiment = Experiment(global_config)
    experiment.create_counterfactuals_table()
    experiment.evaluate_counterfactuals_table()


if __name__ == "__main__":
    run()
    # test()

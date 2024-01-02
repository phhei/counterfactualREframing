# CF4FI

## Counterfactual Explanations for Frame Identification Classifiers

### How to run experiments @pheinisch

- The ``main.py`` has two functions. Use both the first time.
  - ``create_counterfactuals_table``
  - ``evaluate_counterfactuals_table``
- You can define a config in _experiment_configs/mj.yaml_
  - ``mask_strategies`` will run an experiment per strategy
  - ``num_cfs_per_sentence`` is the number of counterfactuals per sentence that will be generated. So far, all are taken
      and evaluated.
    - ``top_k_embedding_similarities`` defines how many neighbors are considered for the embedding similarity.

## Flair SequenceTaggers

SequenceTaggers are used to calculate the validity-score of the counterfactuals. In addition, they are used to automatically determine the relevant spans, see the 'datasets/[topic]_conll_silver_chunk.txt'. The following models were trained:

### Mj = Marijuana

Bases on **roberta-large** (fifth run)

#### Model card

````text
{'flair_version': '0.11.3', 'pytorch_version': '1.11.0+cu113', 'transformers_version': '4.20.1', 'training_parameters': {'base_path': WindowsPath('experiments/exp1_mj/corpora/conll_gold_token/models/roberta-large/run_5'), 'learning_rate': 5e-06, 'mini_batch_size': 8, 'eval_batch_size': None, 'mini_batch_chunk_size': None, 'max_epochs': 50, 'train_with_dev': False, 'train_with_test': False, 'monitor_train': False, 'monitor_test': False, 'main_evaluation_metric': ('micro avg', 'f1-score'), 'scheduler': <class 'flair.optim.LinearSchedulerWithWarmup'>, 'anneal_factor': 0.5, 'patience': 3, 'min_learning_rate': 0.0001, 'initial_extra_patience': 0, 'optimizer': <class 'torch.optim.adamw.AdamW'>, 'cycle_momentum': False, 'warmup_fraction': 0.1, 'embeddings_storage_mode': 'none', 'checkpoint': False, 'save_final_model': True, 'anneal_with_restarts': False, 'anneal_with_prestarts': False, 'anneal_against_dev_loss': False, 'batch_growth_annealing': False, 'shuffle': True, 'param_selection_mode': False, 'write_weights': False, 'num_workers': None, 'sampler': None, 'use_amp': False, 'amp_opt_level': 'O1', 'eval_on_train_fraction': 0.0, 'eval_on_train_shuffle': False, 'save_model_each_k_epochs': 0, 'tensorboard_comment': '', 'use_swa': False, 'use_final_model_for_eval': False, 'gold_label_dictionary_for_eval': None, 'exclude_labels': [], 'create_file_logs': True, 'create_loss_file': True, 'epoch': 40, 'use_tensorboard': False, 'tensorboard_log_dir': None, 'metrics_for_tensorboard': [], 'optimizer_state_dict': None, 'scheduler_state_dict': None, 'save_optimizer_state': False, 'kwargs': {'lr': 5e-06}}}
````

#### Model performance

````text
By class:
                      precision    recall  f1-score   support

HEALTH/PSYCHOLOGICAL     0.7321    0.8831    0.8006       325
  COMMUNITY/SOCIATAL     0.6068    0.5665    0.5859       316
            NATIONAL     0.7913    0.7212    0.7546       226
                DRUG     0.4895    0.6158    0.5455       190
               CHILD     0.6420    0.7123    0.6753       146
             MEDICAL     0.9609    0.7365    0.8339       167
             ILLEGAL     0.6552    0.5327    0.5876       107
             GATEWAY     0.9231    0.7692    0.8392        78
                HARM     0.4211    0.5818    0.4885        55
            PERSONAL     0.8947    0.5862    0.7083        58
               LEGAL     0.7500    0.7674    0.7586        43
           ADDICTION     0.6296    0.9444    0.7556        18

           micro avg     0.6856    0.6975    0.6915      1729
           macro avg     0.7080    0.7014    0.6945      1729
        weighted avg     0.7036    0.6975    0.6940      1729
````

### Mw = minimum wage

Bases on **roberta-large** (forth run)

#### Model card

````text
{'flair_version': '0.12.2', 'pytorch_version': '2.0.1', 'transformers_version': '4.29.2', 'training_parameters': {'base_path': 'argument-aspect-corpus-v1/experiments/exp1_mw/corpora/conll_gold_token/models/roberta-large/run_4', 'learning_rate': 5e-06, 'mini_batch_size': 16, 'eval_batch_size': None, 'mini_batch_chunk_size': None, 'max_epochs': 50, 'train_with_dev': False, 'train_with_test': False, 'monitor_train': False, 'monitor_test': False, 'main_evaluation_metric': ('micro avg', 'f1-score'), 'scheduler': <class 'flair.optim.LinearSchedulerWithWarmup'>, 'anneal_factor': 0.5, 'patience': 3, 'min_learning_rate': 0.0001, 'initial_extra_patience': 0, 'optimizer': <class 'torch.optim.adamw.AdamW'>, 'cycle_momentum': False, 'warmup_fraction': 0.1, 'embeddings_storage_mode': 'none', 'checkpoint': False, 'save_final_model': True, 'anneal_with_restarts': False, 'anneal_with_prestarts': False, 'anneal_against_dev_loss': False, 'batch_growth_annealing': False, 'shuffle': True, 'param_selection_mode': False, 'write_weights': False, 'num_workers': None, 'sampler': None, 'use_amp': False, 'amp_opt_level': 'O1', 'eval_on_train_fraction': 0.0, 'eval_on_train_shuffle': False, 'save_model_each_k_epochs': 0, 'tensorboard_comment': '', 'use_swa': False, 'use_final_model_for_eval': False, 'gold_label_dictionary_for_eval': None, 'exclude_labels': [], 'create_file_logs': True, 'create_loss_file': True, 'epoch': 32, 'use_tensorboard': False, 'tensorboard_log_dir': None, 'metrics_for_tensorboard': [], 'optimizer_state_dict': None, 'scheduler_state_dict': None, 'save_optimizer_state': False, 'reduce_transformer_vocab': False, 'shuffle_first_epoch': False, 'kwargs': {'lr': 5e-06}}}
````

#### Model performance

````text
By class:
                      precision    recall  f1-score   support

              SOCIAL     0.6320    0.4716    0.5402       335
       UN/EMPLOYMENT     0.6795    0.7114    0.6951       149
  MOTIVATION/CHANCES     0.4898    0.5854    0.5333       123
              PRICES     0.6772    0.8515    0.7544       101
COMPETITION/BUSINESS     0.5158    0.5698    0.5414        86
         LOW-SKILLED     0.5647    0.7619    0.6486        63
            ECONOMIC     0.5231    0.6296    0.5714        54
          GOVERNMENT     0.7273    0.5517    0.6275        58
               YOUTH     0.6061    0.5128    0.5556        39
            TURNOVER     0.8710    0.9000    0.8852        30
             CAPITAL     0.3793    0.3667    0.3729        30
             WELFARE     0.3548    0.6875    0.4681        16

           micro avg     0.5984    0.6033    0.6008      1084
           macro avg     0.5850    0.6333    0.5995      1084
        weighted avg     0.6077    0.6033    0.5977      1084
````

### Ab = Abortion

MISSING/ not necessary since this topic was added afterward by the original authors

### Ne = Nuclear Energy

Bases on **roberta-large** (first run)

#### Model card

````text
{'flair_version': '0.12.2', 'pytorch_version': '2.0.1', 'transformers_version': '4.29.2', 'training_parameters': {'base_path': 'argument-aspect-corpus-v1/experiments/exp1_ne/corpora/conll_gold_token/models/roberta-large/run_1', 'learning_rate': 5e-06, 'mini_batch_size': 16, 'eval_batch_size': None, 'mini_batch_chunk_size': None, 'max_epochs': 50, 'train_with_dev': False, 'train_with_test': False, 'monitor_train': False, 'monitor_test': False, 'main_evaluation_metric': ('micro avg', 'f1-score'), 'scheduler': <class 'flair.optim.LinearSchedulerWithWarmup'>, 'anneal_factor': 0.5, 'patience': 3, 'min_learning_rate': 0.0001, 'initial_extra_patience': 0, 'optimizer': <class 'torch.optim.adamw.AdamW'>, 'cycle_momentum': False, 'warmup_fraction': 0.1, 'embeddings_storage_mode': 'none', 'checkpoint': False, 'save_final_model': True, 'anneal_with_restarts': False, 'anneal_with_prestarts': False, 'anneal_against_dev_loss': False, 'batch_growth_annealing': False, 'shuffle': True, 'param_selection_mode': False, 'write_weights': False, 'num_workers': None, 'sampler': None, 'use_amp': False, 'amp_opt_level': 'O1', 'eval_on_train_fraction': 0.0, 'eval_on_train_shuffle': False, 'save_model_each_k_epochs': 0, 'tensorboard_comment': '', 'use_swa': False, 'use_final_model_for_eval': False, 'gold_label_dictionary_for_eval': None, 'exclude_labels': [], 'create_file_logs': True, 'create_loss_file': True, 'epoch': 45, 'use_tensorboard': False, 'tensorboard_log_dir': None, 'metrics_for_tensorboard': [], 'optimizer_state_dict': None, 'scheduler_state_dict': None, 'save_optimizer_state': False, 'reduce_transformer_vocab': False, 'shuffle_first_epoch': False, 'kwargs': {'lr': 5e-06}}}
````

#### Model performance

````text
                    precision    recall  f1-score   support

     ENVIRONMENTAL     0.6417    0.7547    0.6936       159
ACCIDENTS/SECURITY     0.6698    0.4494    0.5379       158
            HEALTH     0.6571    0.6509    0.6540       106
        RENEWABLES     0.8298    0.8764    0.8525        89
            ENERGY     0.4198    0.4928    0.4533        69
       RELIABILITY     0.3855    0.5079    0.4384        63
             WASTE     0.8493    0.9394    0.8921        66
            FOSSIL     0.8286    0.9206    0.8722        63
             COSTS     0.6667    0.5373    0.5950        67
            PUBLIC     0.1667    0.1552    0.1607        58
     TECHNOLOGICAL     0.6129    0.5278    0.5672        36
           WEAPONS     0.7600    0.5278    0.6230        36

         micro avg     0.6303    0.6258    0.6280       970
         macro avg     0.6240    0.6117    0.6117       970
      weighted avg     0.6357    0.6258    0.6239       970
````

## Citation

````bibtex
@inproceedings{heinisch-etal-2023-unsupervised,
    title = "Unsupervised argument reframing with a counterfactual-based approach",
    author = "Heinisch, Philipp  and
      Mindlin, Dimitry  and
      Cimiano, Philipp",
    editor = "Alshomary, Milad  and
      Chen, Chung-Chi  and
      Muresan, Smaranda  and
      Park, Joonsuk  and
      Romberg, Julia",
    booktitle = "Proceedings of the 10th Workshop on Argument Mining",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.argmining-1.11",
    doi = "10.18653/v1/2023.argmining-1.11",
    pages = "107--119",
    abstract = "Framing is an important mechanism in argumentation, as participants in a debate tend to emphasize those aspects or dimensions of the issue under debate that support their standpoint. The task of reframing an argument, that is changing the underlying framing, has received increasing attention recently. We propose a novel unsupervised approach to argument reframing that takes inspiration from counterfactual explanation generation approaches in the field of eXplainable AI (XAI). We formalize the task as a mask-and-replace approach in which an LLM is tasked to replace masked tokens associated with a set of frames to be eliminated by other tokens related to a set of target frames to be added. Our method relies on two key mechanisms: framed decoding and reranking based on a number of metrics similar to those used in XAI to search for a suitable counterfactual. We evaluate our approach on three topics using the dataset by Ruckdeschel and Wiedemann (2022). We show that our two key mechanisms outperform an unguided LLM as a baseline by increasing the ratio of successfully reframed arguments by almost an order of magnitude.",
}
````

from collections import Counter
from typing import List, Optional, Set, Tuple, Dict

import torch
import transformers
import re

from nltk.tokenize.treebank import TreebankWordDetokenizer
from loguru import logger


class T5Editor:
    def __init__(self, model_name: str, **generator_args):
        self.tok = transformers.T5Tokenizer.from_pretrained(model_name)
        self.model = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
        logger.info("Fetched model/tokenizer from HuggingFace model hub ({})", self.tok.name_or_path)
        if torch.cuda.is_available():
            logger.success("CUDA is available, speedy counterfactual creation!")
            self.model.cuda()
            self.device = "cuda"
        else:
            if "large" in model_name:
                logger.warning("CUDA is not available, counterfactual creation will be slow ({})", model_name)
            self.device = "cpu"

        generator_args["max_new_tokens"] = generator_args.get("max_new_tokens", 25)
        generator_args["min_new_tokens"] = generator_args.get("min_new_tokens", 4)
        generator_args["do_sample"] = generator_args.get("do_sample", True)
        generator_args["temperature"] = generator_args.get("temperature", 1.25)
        generator_args["top_k"] = generator_args.get("top_k", 0)
        generator_args["top_p"] = generator_args.get("top_p", 1.)
        # ValueError: Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`.
        # generator_args["diversity_penalty"] = generator_args.get("diversity_penalty", 0.6)
        generator_args["no_repeat_ngram_size"] = generator_args.get("no_repeat_ngram_size", 3)
        generator_args["renormalize_logits"] = generator_args.get("renormalize_logits", True)
        generator_args["remove_invalid_values"] = generator_args.get("remove_invalid_values", True)
        generator_args["output_attentions "] = False
        generator_args["output_hidden_states "] = False
        generator_args["output_scores "] = False
        generator_args["return_dict_in_generate"] = False
        logger.debug("Specified generation arguments: {}", generator_args)
        self.generation_args = transformers.GenerationConfig.from_pretrained(
            model_name, **generator_args
        )
        logger.trace("Generation arguments: {}", self.generation_args)

        self.logits_processor: Optional[Dict[str, transformers.LogitsProcessor]] = None

    def activate_frame_decoding(self, framed_tokens: List[Tuple[str, str]],
                                frame_decoding_strength: float = 0.1) -> None:
        """
        Enables the frame decoding feature of the T5Editor. This feature allows to guide T5 in the generation of
        frame-specific counterfactuals.
        :param framed_tokens: a list of tokens that are labeled with a specific frame (word, frame)
        :param frame_decoding_strength: how much should the frame decoding influence the generation process.
        Negative values will result in frame-avoiding generation, 0 would disable the feature. A value above 0
        (between 0 and 1) will result in frame-guided generation. 1 will mostly supress frame-unlikly tokens.
        :return: nothing
        """
        assert self.tok.name_or_path.startswith("t5"), "Only suitable for T5 so far"
        framed_tokens_counter = Counter(framed_tokens)
        logger.debug("Counter of framed tokens created. The following three combinations are most common: {}",
                     framed_tokens_counter.most_common(n=3))
        frames = set([frame for _, frame in framed_tokens])
        logger.debug("Found {} frames: {}", len(frames), frames)
        vocab_list = [
            (
                word if word.lower() == word.upper() else
                (word.lower().lstrip("\u2581") if word.startswith("\u2581") else f"_{word.lower()}"),
                index
            )
            for word, index in self.tok.get_vocab().items()
        ]
        vocab_list.sort(key=lambda x: x[1], reverse=False)
        vocab_target_size = self.model.config.vocab_size
        logger.debug("Fetched {} of {} vocabs from tokenizer", len(vocab_list), vocab_target_size)

        class FrameLogitsProcessor(transformers.LogitsProcessor):
            def __init__(self, frame: str):
                logger.info("Creating logits processor for frame {}", frame)
                self.frame = frame
                self.frame_decoding_strength = frame_decoding_strength
                self.vocab_distribution: Optional[torch.FloatTensor] = \
                    torch.tensor(
                        data=[framed_tokens_counter.get((vocab, frame), 0)/
                              max(1, sum(map(lambda tf: framed_tokens_counter.get((vocab, tf), 0), frames)))
                              for vocab, _ in vocab_list],
                        dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu", requires_grad=False)
                logger.debug(
                    "Vocab distribution for frame {}: {}... ({}-{})",
                    frame,
                    ", ".join(map(lambda push: str(round(push, 5)), self.vocab_distribution.cpu().tolist()[:10])),
                    torch.min(self.vocab_distribution).cpu().item(),
                    torch.max(self.vocab_distribution).cpu().item()
                )

                if self.vocab_distribution.shape[-1] < vocab_target_size:
                    logger.warning("Padding vocab distribution to {} vocabs because we're too short",
                                   vocab_target_size - self.vocab_distribution.shape[-1])
                    self.vocab_distribution = torch.constant_pad_nd(
                        self.vocab_distribution,
                        (0, vocab_target_size - self.vocab_distribution.shape[-1]),
                        0.0
                    )

            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
                max_distance = (torch.max(scores, dim=-1)[0] - torch.min(scores, dim=-1)[0])
                max_distance = torch.repeat_interleave(
                    torch.unsqueeze(torch.nan_to_num(max_distance, nan=0.0, posinf=10.0, neginf=0.0), dim=1),
                    repeats=vocab_target_size,
                    dim=1
                )
                logger.trace("Ok, strength with {}. Max distance: {}", self.frame_decoding_strength, max_distance)
                return scores + max_distance * self.frame_decoding_strength * self.vocab_distribution

        self.logits_processor = {
            frame: FrameLogitsProcessor(frame)
            for frame in frames
        }
        logger.info("Created {} logits processors (for {})",
                    len(self.logits_processor), ", ".join(self.logits_processor.keys()))

    def generate_counterfactuals(self, original_sentences: List[List[str]],
                                 original_labels: List[List[str]],
                                 target_labels: Set[str],
                                 num_cfs: int = 5,
                                 topic: Optional[str] = None,
                                 other_class_labels: Optional[List[str]] = None) -> List[List[str]]:
        """
        Generates counterfactuals for the given sentences. The counterfactuals are generated by replacing the tokens
        from frames that are not in the target_labels.
        :param original_sentences: the original sentences which should be reframed (tokenized)
        :param original_labels: the original labels of the sentences (aligned with the sentences)
        :param target_labels: the set of labels that should be represented in the counterfactuals
        :param num_cfs: number of counterfactuals to generate (for each sentence)
        :param topic: the topic of the counterfactuals (will be prepended to the counterfactuals).
        If None is given, the counterfactuals are topic-agnostic
        :param other_class_labels: the tokens labeled with that class that should be ignored by the replacer
        (e.g. "O" or "OTHER" - this is assumed if no information is given)
        :return: For each original sentence, a list of counterfactuals (at least tries) is returned.
        May be improved by calling activate_frame_decoding() beforehand.
        """
        if other_class_labels is None:
            other_class_labels = ["O", "OTHER"]
            logger.debug("No information is given regarding other class labels. Assuming {}",
                         " + ".join(other_class_labels))

        reorganized_sentences = []
        number_id = 0
        for sent, label_list in zip(original_sentences, original_labels):
            new_sent = [] if topic is None else ["Regarding", f"{topic},"]
            for token, label in zip(sent, label_list):
                if label in other_class_labels or label in target_labels:
                    logger.trace("Appending {} to reorganized sentence ({})", token, label)
                    new_sent.append(token)
                else:
                    logger.debug("Have to replace \"{}\" ({})", token, label)
                    if len(new_sent) == 0 or not new_sent[-1].startswith("<extra_id"):
                        new_sent.append(f"<extra_id_{number_id}>")
                        number_id += 1

            missing_target_labels = target_labels.difference(label_list)
            if len(missing_target_labels) > number_id:
                logger.debug("Have to add {} more <extra_id> tokens ({})",
                             len(missing_target_labels) - number_id, "|".join(missing_target_labels))
                new_sent.extend([f"<extra_id_{i}>" for i in range(number_id, len(missing_target_labels))])

            logger.debug("Reorganized sentence: {}->{}", sent, new_sent)
            reorganized_sentences.append(new_sent)

        logger.debug("Gathered {} reorganized sentences", len(reorganized_sentences))
        batch = self.tok(text=reorganized_sentences,
                         padding=True,
                         return_tensors="pt",
                         truncation=True,
                         is_split_into_words=True)

        logits_processor = None if self.logits_processor is None else transformers.LogitsProcessorList(
            [lp for frame, lp in self.logits_processor.items() if frame in target_labels]
        )
        logger.trace("Logits processor: {}", logits_processor)

        try:
            suggestions = self.tok.batch_decode(
                sequences=self.model.generate(
                    **batch.to(self.device),
                    generation_config=self.generation_args,
                    logits_processor=logits_processor,
                    num_beams=num_cfs*2, num_return_sequences=num_cfs#,
                    # num_beam_groups=num_cfs*2,
                    # force_words_ids=[[self.tok.get_vocab()[f"<extra_id_{i}>"]] for i in range(number_id)]
                ),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=True
            )
            logger.info("Generated {} suggestions: {}", len(suggestions), suggestions)
        except RuntimeError:
            logger.opt(exception=True).error("Could not generate counterfactuals ({} beams)", num_cfs*2)
            return [["RUNTIME-ERROR"]*num_cfs]*len(original_sentences)

        counterfactuals = []
        counterfactuals_for_single_sentence = []
        detokenizer = TreebankWordDetokenizer()
        for i, sug in enumerate(suggestions):
            if i % num_cfs == 0 and i > 0:
                counterfactuals.append(counterfactuals_for_single_sentence)
                counterfactuals_for_single_sentence = []

            counterfactual = reorganized_sentences[int(i/num_cfs)].copy()
            if topic is not None:
                counterfactual = counterfactual[2:]

            for extra_id_str in re.findall(pattern=r"\<extra\_id\_\d+\>", string=" ".join(counterfactual)):
                extra_id = int(re.search(pattern=r"\d+", string=extra_id_str)[0])

                def extract_from_suggestion() -> str:
                    try:
                        span = sug[sug.index(extra_id_str)+len(extra_id_str):sug.index(f"<extra_id_{extra_id+1}>") if f"<extra_id_{extra_id+1}>" in sug else len(sug)].strip()
                        for special_token in self.tok.all_special_tokens:
                            span = span.replace(special_token, "")
                        return span
                    except ValueError:
                        logger.opt(exception=True).warning("Could not extract {} from suggestion: {}",
                                                           extra_id_str, sug)
                        return ""
                counterfactual = [extract_from_suggestion() if token == extra_id_str else token for token in counterfactual]

            logger.trace("Counterfactual: {}", counterfactual)
            counterfactuals_for_single_sentence.append(detokenizer.detokenize(counterfactual))
            logger.debug("Added detokenized counterfactual: {}", counterfactuals_for_single_sentence[-1])

        counterfactuals.append(counterfactuals_for_single_sentence)
        logger.success("Generated {} counterfactuals", sum(map(len, counterfactuals)))

        return counterfactuals

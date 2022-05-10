import code
import math
import random

import torch
from torch import nn

from models.embeddings.scalar_mix import ScalarMixWithDropout
from models.embeddings.wrappers import Wrapper

from transformers.tokenization_bart import BartTokenizer
from transformers.modeling_bart import BartModel, BartConfig

from transformers.tokenization_mbart import MBartTokenizer
from transformers.modeling_mbart import MBartConfig

MBART_LANG_CODES = {"ar": "ar_AR",
                    "cs": "cs_CZ",
                    "de": "de_DE",
                    "en": "en_XX",
                    "es": "es_XX",
                    "et": "et_EE",
                    "fi": "fi_FI",
                    "fr": "fr_XX",
                    "gu": "gu_IN",
                    "hi": "hi_IN",
                    "it": "it_IT",
                    "ja": "ja_XX",
                    "kk": "kk_KZ",
                    "ko": "ko_KR",
                    "lt": "lt_LT",
                    "lv": "lv_LV",
                    "my": "my_MM",
                    "ne": "ne_NP",
                    "nl": "nl_XX",
                    "ro": "ro_RO",
                    "ru": "ru_RU",
                    "si": "si_LK",
                    "tr": "tr_TR",
                    "vi": "vi_VN",
                    "zh": "zh_CN"}


class BartBaseWrapper(Wrapper):
    def __init__(self, model_class, tokenizer_class, config_class, model_path, output_ids, tokenizer_path=None,
                 config_only=False, fine_tune=False, hidden_dropout=0.2, attn_dropout=0.2, output_dropout=0.5,
                 scalar_mix_layer_dropout=0.1, token_mask_prob=0.2, language=None, poisson_lambda=3):
        super(BartBaseWrapper, self).__init__(model_class, tokenizer_class, config_class, model_path, output_ids,
                                          tokenizer_path=tokenizer_path, config_only=config_only, fine_tune=fine_tune,
                                          hidden_dropout=hidden_dropout, attn_dropout=attn_dropout, output_dropout=output_dropout,
                                          scalar_mix_layer_dropout=scalar_mix_layer_dropout, token_mask_prob=token_mask_prob)

        self.language_code = None if language is None else MBART_LANG_CODES[language]

        if poisson_lambda > 0:
            self.mask_distribution = get_poisson(poisson_lambda)
        else:
            self.mask_distribution = None

    def _init_model(self, model_class, tokenizer_class, config_class, model_path, tokenizer_path, config_only=False,
                    hidden_dropout=0.2, attn_dropout=0.2):
        if config_only:
            model = model_class(config_class.from_json_file(str(model_path)))
            tokenizer = tokenizer_class.from_pretrained(str(tokenizer_path))
        else:
            model = model_class.from_pretrained(model_path,
                                                output_hidden_states=True,
                                                dropout=hidden_dropout,
                                                attention_dropout=attn_dropout)
            tokenizer = tokenizer_class.from_pretrained(model_path)

        return model, tokenizer

    def _init_scalar_mix(self, layer_dropout=0.1):
        num_layers = len(self.model.decoder.layers)
        scalar_mix = nn.ModuleDict(
            {output_id: ScalarMixWithDropout(mixture_size=num_layers, layer_dropout=layer_dropout) for
             output_id in self.output_ids})

        return scalar_mix

    def _get_model_inputs(self, input_sentences):
        """Take a list of sentences and return tensors for token IDs, attention mask, and original token mask"""
        mask_ratio = self.token_mask_prob if self.training else 0.0
        input_sequences = [BartInputSequence(sent, self.tokenizer, language_code=self.language_code,
                                             mask_ratio=mask_ratio, mask_distribution=self.mask_distribution)
                           for sent in input_sentences]
        max_encoder_seq_len = max(len(input_sequence.encoder_token_ids) for input_sequence in input_sequences)
        max_decoder_seq_len = max(len(input_sequence.decoder_token_ids) for input_sequence in input_sequences)
        device = next(iter(self.model.parameters())).device  # Ugly :(
        for input_sequence in input_sequences:
            input_sequence.tensorize(device, encoder_padded_length=max_encoder_seq_len, decoder_padded_length=max_decoder_seq_len)

        # Batch components of input sequences
        encoder_input_ids = torch.stack([input_seq.encoder_token_ids for input_seq in input_sequences])
        encoder_attention_mask = torch.stack([input_seq.encoder_attention_mask for input_seq in input_sequences])

        decoder_input_ids = torch.stack([input_seq.decoder_token_ids for input_seq in input_sequences])
        decoder_attention_mask = torch.stack([input_seq.decoder_attention_mask for input_seq in input_sequences])
        decoder_original_token_mask = torch.stack([input_seq.decoder_orig_token_mask for input_seq in input_sequences])

        assert encoder_input_ids.shape[0] == decoder_input_ids.shape[0] == len(input_sentences)
        assert encoder_input_ids.shape[1] == max_encoder_seq_len and decoder_input_ids.shape[1] == max_decoder_seq_len

        return {"encoder_input_ids": encoder_input_ids, "encoder_attention_mask": encoder_attention_mask,
                "decoder_input_ids": decoder_input_ids, "decoder_attention_mask": decoder_attention_mask,
                "original_token_mask": decoder_original_token_mask, "device": device}

    def _get_raw_embeddings(self, model_inputs):
        """Take tensors for input tokens and run them through underlying BERT-based model, performing the learned scalar
         mixture for each output"""
        raw_embeddings = dict()

        encoder_input_ids = model_inputs["encoder_input_ids"]
        encoder_attention_mask = model_inputs["encoder_attention_mask"]
        decoder_input_ids = model_inputs["decoder_input_ids"]
        decoder_attention_mask = model_inputs["decoder_attention_mask"]

        with torch.set_grad_enabled(self.fine_tune):
            model_output = self.model(input_ids=encoder_input_ids, attention_mask=encoder_attention_mask,
                                      decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask,
                                      use_cache=False, return_dict=True)
            embedding_layers = torch.stack(model_output["decoder_hidden_states"])
            for output_id in self.output_ids:
                if self.output_dropout:
                    embedding_layers_with_dropout = self.output_dropout(embedding_layers)
                curr_output = self.scalar_mix[output_id](embedding_layers_with_dropout)
                raw_embeddings[output_id] = curr_output

        return raw_embeddings


class BartWrapper(BartBaseWrapper):
    def __init__(self, *args, **kwargs):
        super(BartWrapper, self).__init__(BartModel, BartTokenizer, BartConfig, *args, **kwargs)


class MBartWrapper(BartBaseWrapper):
    def __init__(self, *args, **kwargs):
        super(MBartWrapper, self).__init__(BartModel, MBartTokenizer, MBartConfig, *args, **kwargs)


class BartInputSequence:
    """Class for representing the features of a single, dependency-annotated sentence in tensor
       form, for usage in models based on BART.
    """
    def __init__(self, orig_tokens, tokenizer, language_code=None, mask_ratio=0.0, mask_distribution=None):
        self.tokenizer = tokenizer
        self.language_code = language_code
        self.mask_ratio = mask_ratio
        self.mask_distribution = mask_distribution

        self.tokens = list()
        self.original_token_mask = list()

        self.prepare_common_token_seq(orig_tokens)
        self.prepare_encoder_input()
        self.prepare_decoder_input()

    def prepare_common_token_seq(self, orig_tokens):
        if self.language_code is not None:
            self.append_special_token(self.language_code)
        else:
            self.append_special_token(self.tokenizer.bos_token)

        for orig_token in orig_tokens:
            self.append_regular_token(orig_token)

        self.append_special_token(self.tokenizer.eos_token)

        if self.language_code is not None:
            self.append_special_token(self.language_code)

    def prepare_encoder_input(self):
        self.encoder_tokens = self.tokens[1:]
        self.encoder_orig_token_mask = self.original_token_mask[1:]

        if self.mask_ratio == 0.0:
            self.encoder_token_ids = self.tokenizer.convert_tokens_to_ids(self.encoder_tokens)
            self.encoder_attention_mask = [1] * len(self.encoder_token_ids)
            return

        if self.mask_distribution is None:
            self.prepare_encoder_input_simple_masking()
        else:
            self.prepare_encoder_input_text_infilling()

        self.encoder_token_ids = self.tokenizer.convert_tokens_to_ids(self.encoder_tokens)
        self.encoder_attention_mask = [1] * len(self.encoder_token_ids)

    def prepare_encoder_input_simple_masking(self):
        currently_masking = False
        for i, is_orig in enumerate(self.encoder_orig_token_mask):
            if self.encoder_tokens[i] in set(MBART_LANG_CODES.values()) |\
                    {self.tokenizer.bos_token, self.tokenizer.eos_token}:
                # Never mask special tokens
                continue

            if is_orig:
                if random.random() < self.mask_ratio:
                    currently_masking = True
                    self.encoder_tokens[i] = self.tokenizer.mask_token
                else:
                    currently_masking = False
            else:
                if currently_masking:
                    self.encoder_tokens[i] = self.tokenizer.mask_token

    def prepare_encoder_input_text_infilling(self):
        num_to_mask = int(math.ceil(sum(self.encoder_orig_token_mask) * self.mask_ratio))
        if num_to_mask == 0:
            return

        # Sample span lengths from distribution
        lengths = self.mask_distribution.sample(sample_shape=(num_to_mask,))

        # Make sure we have enough to mask
        cum_length = torch.cumsum(lengths, 0)
        while cum_length[-1] < num_to_mask:
            lengths = torch.cat([lengths, self.mask_distribution.sample(sample_shape=(num_to_mask,))], dim=0)
            cum_length = torch.cumsum(lengths, 0)

        # Trim to masking budget
        i = 0
        while cum_length[i] < num_to_mask:
            i += 1
        lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i-1])

        num_spans = i + 1
        lengths = lengths[:num_spans]

        # Handle 0-length mask (inserts) separately
        lengths = lengths[lengths > 0]
        num_inserts = num_spans - lengths.size(0)
        num_spans -= num_inserts
        if num_spans == 0:
            self.add_insertion_noise(num_inserts)
            return

        assert (lengths > 0).all()

        # Get random start points of spans
        word_starts = [i for i, m in enumerate(self.encoder_orig_token_mask) if m > 0]
        span_starts = set(random.sample(word_starts, num_spans))

        # Iterate over tokens and mask according to starting points and span length
        lengths = list(lengths)
        curr_mask_budget = 0
        currently_masking = False
        for i, is_orig in enumerate(self.encoder_orig_token_mask):
            if self.encoder_tokens[i] in set(MBART_LANG_CODES.values()) | \
                    {self.tokenizer.bos_token, self.tokenizer.eos_token}:
                # Never mask special tokens
                continue

            if is_orig and curr_mask_budget == 0:
                currently_masking = False

            if i in span_starts:
                curr_mask_budget = max(curr_mask_budget, lengths.pop())
                currently_masking = True

            if currently_masking:
                self.encoder_tokens[i] = self.tokenizer.mask_token
                if is_orig:
                    curr_mask_budget -= 1

        # Add insertion noise
        if num_inserts > 0:
            self.add_insertion_noise(num_inserts)

        # Iterate over tokens a second time and collapse adjacent MASK tokens into a single MASK token
        reduced_tokens = list()
        within_mask = False
        for i, token in enumerate(self.encoder_tokens):
            if token == self.tokenizer.mask_token:
                if within_mask:
                    pass
                else:
                    within_mask = True
                    reduced_tokens.append(self.tokenizer.mask_token)
            else:
                within_mask = False
                reduced_tokens.append(token)

        self.encoder_tokens = reduced_tokens

    def add_insertion_noise(self, num_inserts):
        if num_inserts == 0:
            return

        word_starts = [i for i, m in enumerate(self.encoder_orig_token_mask) if m > 0 and self.encoder_tokens[i] not in MBART_LANG_CODES.values()]
        num_inserts = min(num_inserts, len(word_starts))  # Cannot insert more noise than word starts
        insertion_indices = sorted(random.sample(word_starts, num_inserts), reverse=True)

        for ix in insertion_indices:
            self.encoder_tokens.insert(ix, self.tokenizer.mask_token)

    def prepare_decoder_input(self):
        self.decoder_tokens = self.tokens[:-1]
        self.decoder_orig_token_mask = self.original_token_mask[:-1]
        self.decoder_token_ids = self.tokenizer.convert_tokens_to_ids(self.decoder_tokens)
        self.decoder_attention_mask = [1] * len(self.decoder_token_ids)

    def append_special_token(self, token):
        """Append a special token (e.g. BOS token, MASK token) to the sequence. The token will not be counted as
        an original token.
        """
        self.tokens.append(token)
        self.original_token_mask.append(0)

    def append_regular_token(self, token):
        """Append regular token (i.e., a word from the input sentence) to the sequence. The token will be split further
        into word pieces by the tokenizer. All word pieces will receive attention, but only the first word piece will
        be counted as an original token."""
        curr_bart_tokens = self.tokenizer.tokenize(token)

        assert len(curr_bart_tokens) > 0

        self.tokens += curr_bart_tokens
        self.original_token_mask += [1] + [0] * (len(curr_bart_tokens) - 1)

    def tensorize(self, device, encoder_padded_length=None, decoder_padded_length=None):
        assert (encoder_padded_length is None) == (decoder_padded_length is None)

        if encoder_padded_length is not None:
            self.pad_to_length(encoder_padded_length, decoder_padded_length)

        self.encoder_token_ids = torch.tensor(self.encoder_token_ids, device=device)
        self.encoder_attention_mask = torch.tensor(self.encoder_attention_mask, device=device)

        self.decoder_token_ids = torch.tensor(self.decoder_token_ids, device=device)
        self.decoder_attention_mask = torch.tensor(self.decoder_attention_mask, device=device)

        self.decoder_orig_token_mask = torch.tensor(self.decoder_orig_token_mask, device=device)

    def pad_to_length(self, encoder_padded_length, decoder_padded_length):
        """Pad the sentence to the specified length. This will increase the length of all fields to padded_length by
        adding the padding label/index."""
        assert encoder_padded_length >= len(self.encoder_token_ids)
        assert decoder_padded_length >= len(self.decoder_token_ids)

        encoder_padding_length = encoder_padded_length - len(self.encoder_token_ids)
        decoder_padding_length = decoder_padded_length - len(self.decoder_token_ids)

        self.encoder_tokens += [self.tokenizer.pad_token] * encoder_padding_length
        self.encoder_token_ids += [self.tokenizer.pad_token_id] * encoder_padding_length
        self.encoder_attention_mask += [0] * encoder_padding_length

        assert len(self.encoder_tokens) == len(self.encoder_token_ids) == len(self.encoder_attention_mask)

        self.decoder_tokens += [self.tokenizer.pad_token] * decoder_padding_length
        self.decoder_token_ids += [self.tokenizer.pad_token_id] * decoder_padding_length
        self.decoder_attention_mask += [0] * decoder_padding_length
        self.decoder_orig_token_mask += [0] * decoder_padding_length

        assert len(self.decoder_tokens) == len(self.decoder_token_ids) == len(self.decoder_attention_mask) == len(self.decoder_orig_token_mask)


def get_poisson(_lambda):
    lambda_to_the_k = 1
    e_to_the_minus_lambda = math.exp(-_lambda)

    k_factorial = 1
    ps = []
    for k in range(0, 128):
        ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
        if ps[-1] < 0.0000001:
            break

        lambda_to_the_k *= _lambda
        k_factorial *= (k + 1)

    ps = torch.FloatTensor(ps)
    return torch.distributions.Categorical(ps)


if __name__ == "__main__":
    mbart_wrapper = BartWrapper(BartModel, MBartTokenizer, MBartConfig, "/home/gst2rng/Documents/parsing_data/pretrained_models/mbart-large-cc25",
                                {"heads", "labels"}, language="de")

    #mbart_tokenizer = MBartTokenizer.from_pretrained("/home/gst2rng/Documents/parsing_data/pretrained_models/mbart-large-cc25")

    sent1 = "Ich wollte mir eine Krone machen lassen und war deshalb bei zwei Zahnärzten , die beide unverschämte Preise dafür haben wollten .".split(" ")
    sent2 = "Ich bin dagegen , daß wir alle Erfahrungen der letzten 30 Jahre über Bord werfen .".split(" ")
    sent3 = "Viele Fehler wurden gegen 15 Uhr und die allermeisten Fehler wurden gegen drei Uhr nachts registriert .".split(" ")
    sents = [sent1, sent2, sent3]

    embeddings, true_seq_lengths = mbart_wrapper(sents)

    code.interact(local=locals())
    """
    mask_distribution = get_poisson(3)

    seq = BartInputSequence(orig_tokens, mbart_tokenizer, mask_ratio=0.3, language_code=lang_code, mask_distribution=mask_distribution)
    seq.tensorize(encoder_padded_length=40, decoder_padded_length=40, device="cuda:0")

    print(seq.encoder_tokens)
    print(seq.encoder_token_ids)
    print(seq.encoder_attention_mask)

    print(seq.decoder_tokens)
    print(seq.decoder_token_ids)
    print(seq.decoder_orig_token_mask)
    print(seq.decoder_attention_mask)
    """

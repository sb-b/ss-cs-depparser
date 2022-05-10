import argparse
import io

from argparse import Namespace
from pathlib import Path

from init_config import ConfigParser
from data_handling.custom_conll_dataset import CustomCoNLLDataset
from util.conll18_ud_eval import evaluate_basic
from util.iwpt20_xud_eval import evaluate_enhanced


def parse_corpus(config, corpus_file, output, parser=None, keep_columns=None):
    """Parse each line of the (raw) input corpus. Assumes whitespace tokenization and one sentence per line.
    Can pass either a config (in which the parser to be evaluated will be initialized from this config) or a MultiParser
    object directly.
    """
    if parser is None:
        model = config.init_model()
        trainer = config.init_trainer(model, None, None)  # Inelegant, but need to do this because trainer handles checkpoint loading
        parser = trainer.parser

    annotation_layers = config["data_loaders"]["args"]["annotation_layers"]
    if keep_columns is not None:
        for col in keep_columns:
            annotation_layers[col] = {"type": "TagSequence", "source_column": col}
    column_mapping = {annotation_id: annotation_layer["source_column"] for annotation_id, annotation_layer in annotation_layers.items()}

    dataset = CustomCoNLLDataset.from_corpus_file(corpus_file, annotation_layers)
    for sentence in dataset:
        parsed_sentence = parser.parse(sentence)
        for col in keep_columns or []:  # Copy over columns to keep from input corpus
            parsed_sentence.annotation_data[col] = sentence[col]
        print(parsed_sentence.to_conll(column_mapping), file=output)


def get_config_modification(args, lstm=False):
    """Modify config for parsing/evaluation."""
    modification = dict()

    modification["saving"] = False  # Do not save config

    # Overwrite vocab file paths with saved vocab files in model config directory
    model_dir = Path(args.model_dir)
    for vocab_path in model_dir.glob("*.vocab"):
        outp_id = vocab_path.stem
        modification[f"model.args.outputs.{outp_id}.args.vocab.args.vocab_filename"] = str(vocab_path)

    # Overwrite transformer model configuration file with stored config; do not load weights
    # TODO: Workaround for LSTM, fix this!
    if lstm:
        modification["model.args.embeddings_processor.args.embeddings_wrapper.args.model_path"] = model_dir / "transformer.json"
        modification["model.args.embeddings_processor.args.embeddings_wrapper.args.tokenizer_path"] = model_dir / "tokenizer"
        modification["model.args.embeddings_processor.args.embeddings_wrapper.args.config_only"] = True
    else:
        modification["model.args.embeddings_processor.args.model_path"] = model_dir / "transformer.json"
        modification["model.args.embeddings_processor.args.tokenizer_path"] = model_dir / "tokenizer"
        modification["model.args.embeddings_processor.args.config_only"] = True

    return modification


def create_output(output_filename):
    if not output_filename:
        return io.StringIO()
    else:
        return open(output_filename, "w")


def reset_file(output_file, output_filename):
    """Reset output in read mode"""
    if not output_filename:
        assert isinstance(output_file, io.StringIO)
        output_file.seek(0)
        return output_file
    else:
        assert not isinstance(output_file, io.StringIO)
        output_file.close()
        output_file = open(output_filename, "r")
        return output_file


def run_evaluation(corpus_file, system_file, mode):
    if mode == "basic":
        eval_args = Namespace(
            **{"gold_file": corpus_file, "system_file": system_file, "verbose": True, "counts": False})
        return evaluate_basic(eval_args)
    elif mode == "enhanced":
        eval_args = Namespace(
            **{"gold_file": corpus_file, "system_file": system_file, "verbose": True, "counts": False,
               "enhancements": "0"})
        return evaluate_enhanced(eval_args)
    else:
        raise Exception(f"Unknown evaluation mode {mode}.")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Graph-based UD parser (corpus parsing mode)')

    # Required arguments
    argparser.add_argument('model_dir', type=str, help='path to model directory (required)')
    argparser.add_argument('corpus_filename', type=str, help='path to corpus file (required).')

    # Optional arguments
    argparser.add_argument('-o', '--output-filename', type=str, default="", help='output filename. If none is provided,'
                                                                                 'output will not be saved to disk.')
    argparser.add_argument('-e', '--eval', type=str, default="none", help='Evaluation type (basic/enhanced/none).'
                                                                          'Default: none')
    argparser.add_argument('-k', '--keep-columns', nargs='+', type=int, help='Indices of columns to retain from input'
                                                                             'corpus')

    # TODO: Workaround for LSTM, fix this!
    argparser.add_argument('--lstm', action='store_true', help='Use this flag if model has an LSTM')

    args = argparser.parse_args()
    config = ConfigParser.from_args(args, modification=get_config_modification(args, lstm=args.lstm))
    output_file = create_output(args.output_filename)

    with open(args.corpus_filename, "r") as corpus_file:
        parse_corpus(config, corpus_file, output_file, keep_columns=args.keep_columns)

    # Run evaluation
    if args.eval.lower() not in {"", "none"}:
        output_file = reset_file(output_file, args.output_filename)
        with open(args.corpus_filename, "r") as corpus_file:
            run_evaluation(corpus_file, output_file, args.eval.lower())

    output_file.close()

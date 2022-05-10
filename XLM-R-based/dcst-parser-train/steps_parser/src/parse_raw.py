import argparse
import stanza

from io import StringIO
from stanza.utils.conll import CoNLL

from init_config import ConfigParser
from parse_corpus import parse_corpus, get_config_modification, create_output, reset_file, run_evaluation


def preprocess_to_stream(corpus_filename, lang):
    stanza_pipeline = stanza.Pipeline(lang=lang, processors='tokenize,mwt', use_gpu=False)
    with open(corpus_filename, "r") as corpus_file:
        doc = stanza_pipeline(corpus_file.read())

    conll = CoNLL.convert_dict(doc.to_dict())
    conll_stream = StringIO()
    for sent in conll:
        for token in sent:
            print("\t".join(token), file=conll_stream)
        print(file=conll_stream)

    conll_stream.seek(0)

    return conll_stream


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Graph-based UD parser (raw text parsing mode)')

    # Required arguments
    argparser.add_argument('model_dir', type=str, help='path to model directory (required)')
    argparser.add_argument('language', type=str, help='language code (for tokenization and segmentation).')
    argparser.add_argument('corpus_filename', type=str, help='path to corpus file (required).')

    # Optional arguments
    argparser.add_argument('-o', '--output-filename', type=str, default="", help='output filename. If none is provided,'
                                                                                 'output will not be saved to disk.')
    argparser.add_argument('-r', '--reference-corpus', type=str, default="", help='Reference corpus to evaluate against.')
    argparser.add_argument('-e', '--eval', type=str, default="basic", help='Evaluation type (basic/enhanced).'
                                                                           'Default: basic')
    args = argparser.parse_args()
    config = ConfigParser.from_args(args, modification=get_config_modification(args))
    conll_stream = preprocess_to_stream(args.corpus_filename, args.language)
    output_file = create_output(args.output_filename)

    parse_corpus(config, conll_stream, output_file)

    # Run evaluation
    if args.reference_corpus != "":
        output_file = reset_file(output_file, args.output_filename)
        with open(args.reference_corpus, "r") as reference_corpus:
            run_evaluation(reference_corpus, output_file, args.eval.lower())

    conll_stream.close()

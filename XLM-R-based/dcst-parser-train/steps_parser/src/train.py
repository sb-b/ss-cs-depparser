#  Copyright (c) 2020 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald

"""Main script for training a parser based on a configuration file."""

import argparse

from pathlib import Path

from init_config import ConfigParser
from parse_corpus import reset_file, parse_corpus, run_evaluation

#mycode
import os, psutil
from pynvml import *
import torch
#mycode

class LinearWeightedAvg(torch.nn.Module):
    def __init__(self, n_inputs):
        super(LinearWeightedAvg, self).__init__()
        self.linear = torch.nn.Linear(n_inputs * 768, 768).cuda()
        

    def forward(self, input1, input2):

     

        alpha = torch.sigmoid(self.linear(torch.cat((input1,input2), dim=-1).cuda())) 

        output = torch.mul(alpha, input1) + torch.mul(1 - alpha, input2)

        return output


def main(config, eval_mode="basic"): #mycode
    """Main function to initialize model, load data, and run training.

    Args:
        config: Experimental configuration.
        eval_mode: Method to use in post-training evaluation: "basic" for basic UD, "enhanced" for enhanced UD.
          Default: "basic".
    """
    model = config.init_model()

    data_loaders = config.init_data_loaders(model)

    trainer = config.init_trainer(model, data_loaders["train"], data_loaders["dev"])

    seq_model = config.init_seqmodel()

    seq_data_loaders = config.init_seq_data_loaders(seq_model)
   
    seq_trainer = config.init_seqtrainer(seq_model, seq_data_loaders["train"], seq_data_loaders["dev"])
 
    del seq_data_loaders
    del data_loaders
    del seq_model
    del model

#QTD_SAGT

    seq_trainer._resume_checkpoint("/mount/projekte18/codeswitch/betul/ss-cs-depparser/XLM-R-based/auxiliary-task-train/trained_models/dcst_head_lang_id_trde90_parsedbyqtd/seq_labelling/1207_110423/model_best.pth")


    trainer.parser.seq_embed = seq_trainer.parser.embed
    trainer.parser.seq_embed.requires_grad = False

    trainer.parser.wa_layer = LinearWeightedAvg(2)

    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    print(f'total    : {info.total}')
    print(f'free     : {info.free}')
    print(f'used     : {info.used}')

    
    process = psutil.Process(os.getpid())
    print("MEM USED: ",process.memory_info().rss)  # in bytes 

    #trainer.dcst_train()
    del seq_trainer

    if "test" in config["data_loaders"]["paths"]:
        evaluate_best_trained_model(trainer, config, eval_mode=eval_mode)


def evaluate_best_trained_model(trainer, config, eval_mode="basic"):
    """Evaluate the model with best validation performance on test data after training.

    Args:
        trainer: Trainer used for training the model.
        config: Model configuration (must contain path to test data).
        eval_mode: Method to use in evaluation: "basic" for basic UD, "enhanced" for enhanced UD. Default: "basic".
    """
    checkpoint_path = "/mount/projekte18/codeswitch/betul/ss-cs-depparser/XLM-R-based/dcst-parser-train/trained_models/qtd_sagt_2.8_gating_with_trde90_unlabeled/deps_and_hli_batchsize_16_es_100/1207_125541/model_best.pth"
    #checkpoint_path = Path(trainer.checkpoint_dir) / "model_best.pth"
    trainer._resume_checkpoint(checkpoint_path)

    logger = config.logger

    logger.info("Evaluation on test set:")

    with open(config["data_loaders"]["paths"]["test"], "r") as gold_test_file, \
         open("test-parsed.conllu", "w") as output_file:
        parse_corpus(config, gold_test_file, output_file, parser=trainer.parser)
        output_file = reset_file(output_file, "test-parsed.conllu")
        gold_test_file = reset_file(gold_test_file, config["data_loaders"]["paths"]["test"])
        test_evaluation = run_evaluation(gold_test_file, output_file, mode=eval_mode)

    if eval_mode == "basic":
        logger.log_final_metrics_basic(test_evaluation, suffix="_test")
    elif eval_mode == "enhanced":
        logger.log_final_metrics_enhanced(test_evaluation, suffix="_test")
    else:
        raise Exception(f"Unknown evaluation mode {eval_mode}")

    logger.log_artifact("test-parsed.conllu")


def init_config_modification(raw_modifications):
    """Turn a "raw" config modification string into a dictionary of key-value pairs to replace."""
    modification = dict()
    for mod in raw_modifications:
        key, value = mod.split("=", 1)

        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                if value == "true":
                    value = True
                elif value == "false":
                    value = False

        modification[key] = value

    return modification


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Graph-based enhanced UD parser (training mode)')
    argparser.add_argument('config', type=str, help='config file path (required)')
   # argparser.add_argument('seq_config', type=str, help='config file path (required)') #mycode

    argparser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    argparser.add_argument('-s', '--save-dir', default=None, type=str, help='model save directory (config override)')
    argparser.add_argument('-m', '--modification', default=None, type=str, nargs='+', help='modifications to make to'
                                                                                           'specified configuration file'
                                                                                           '(config override)')
    argparser.add_argument('-e', '--eval', type=str, default="basic", help='Evaluation type (basic/enhanced).'
                                                                           'Default: basic')
    args = argparser.parse_args()

    modification = init_config_modification(args.modification) if args.modification is not None else dict()
    if args.save_dir is not None:
        modification["trainer.save_dir"] = args.save_dir

    config = ConfigParser.from_args(args, modification=modification)
    main(config, eval_mode=args.eval)

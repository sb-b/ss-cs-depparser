# Semi-Supervised CS Dependency Parser

This page includes source codes and trained models of the semi-supervised deep dependency parser described in our paper named ["Improving Code-Switching Dependency Parsing with Semi-Supervised Auxiliary Tasks"](https://2022.naacl.org/program/accepted_papers/#findings). The parser employs a semi-supervised learning approach ["DCST"](https://rotmanguy.github.io/publications/2019-10-01-deep-contextualized-self-learning) and utilizes auxiliary tasks for dependency parsing of code-switched (CS) language pairs. There are two versions of the parsing model, one is LSTM-based and the other is XLM-R-based. The following sections explain how to run these models. The trained models can be found [here](https://drive.google.com/drive/folders/12F4ieakslvFZtOAj4JOqRX3NOTLPLICs?usp=sharing).

## How-To-Run the LSTM-based Parser

### Requirements

Run the following:

    - pip install -r requirements.txt
    
### Datasets

* Navigate to the **LSMT-based/DCST** folder

* Download CS UD Treebanks ["Frisian-Dutch FAME"](https://universaldependencies.org/treebanks/qfn_fame/index.html), ["Hindi-English HIENCS"](https://universaldependencies.org/treebanks/qhe_hiencs/index.html) ["Komi-Zyrian IKDP"](https://universaldependencies.org/treebanks/kpv_ikdp/index.html), and ["Turkish-German SAGT"](https://universaldependencies.org/treebanks/qtd_sagt/index.html) from https://universaldependencies.org and locate them under **data/datasets**

* Download the unlabeled data to be used, convert them to ["CoNLL-U format"](https://universaldependencies.org/format.html) and locate them under **data/datasets**
* Run the script:

    python utils/io_/convert_ud_to_onto_format.py --ud_data_path data/datasets

### Word Embeddings

The LSTM-based models need pretrained word embeddings.

* Download FastText embeddings from https://fasttext.cc/docs/en/crawl-vectors.html

   - In the paper, I used Dutch embeddings for Frisian-Dutch language pair, Hindi embeddings for Hindi-English, Russian embeddings for Komi-Zyrian, and Turkish embeddings for Turkish-German.

* Unzip and locate them under **data/multilingual_word_embeddings** folder

--------------

Let's say we want to train the LSTM-based model with auxiliary task enhancements for the Turkish-German SAGT Treebank (qtd_sagt). As the unlabeled data, we use ["TuGeBiC"](https://github.com/ozlemcek/TuGeBiC) (qtd_trde90).

   - Download the corpus, join all conll-u files and divide them to train and dev files. Name the training as "qtd_trde90-ud-train.conllu" and dev as "qtd_trde90-ud-dev.conllu". Locate these files under the folder **data/datasets/UD_QTD-TRDE90/**

* Run the script:

    python utils/io_/convert_ud_to_onto_format.py --ud_data_path data/datasets

##### 1.1- Train the baseline parser:
    python examples/GraphParser.py --dataset ud --domain qtd_sagt --rnn_mode LSTM --num_epochs 150 --batch_size 16 --hidden_size 512 --arc_space 512 --arc_tag_space 128 --num_layers 3 --num_filters 100 --use_char --use_pos --word_dim 300 --char_dim 100 --pos_dim 100 --initializer xavier --opt adam --learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 --p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --arc_decode mst --unk_replace 0.5 --punct_set '.' '``'  ':' ','  --word_embedding fasttext --word_path "data/multilingual_word_embeddings/cc.tr.300.vec" --char_embedding random --model_path saved_models/ud_parser_qtd_sagt_full_train

##### 1.2- Parse the unlabeled data:
    - python examples/GraphParser.py --dataset ud --domain qtd_trde90 --rnn_mode LSTM --num_epochs 150 --batch_size 16 --hidden_size 512 --arc_space 512 --arc_tag_space 128 --num_layers 3 --num_filters 100 --use_char --use_pos --word_dim 300 --char_dim 100 --pos_dim 100 --initializer xavier --opt adam --learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 --p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --arc_decode mst --unk_replace 0.5 --punct_set '.' '``'  ':' ','  --word_embedding fasttext --word_path "data/multilingual_word_embeddings/cc.tr.300.vec" --char_embedding random --model_path saved_models/ud_parser_qtd_sagt_full_train --eval_mode --strict --load_path saved_models/ud_parser_qtd_sagt_full_train/domain_qtd_sagt.pt

##### 1.3- Train sequence labelers:

###### Number of Children Task (NOC):
    - python examples/SequenceTagger_for_DA.py --dataset ud --src_domain qtd_sagt --tgt_domain qtd_trde90 --task number_of_children --rnn_mode LSTM --num_epochs 100 --batch_size 16 --hidden_size 512 --tag_space 128 --num_layers 3 --num_filters 100 --use_char  --use_pos --char_dim 100 --pos_dim 100 --initializer xavier --opt adam --learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 --p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --unk_replace 0.5 --punct_set '.' '``'  ':' ','  --word_embedding fasttext --word_path "data/multilingual_word_embeddings/cc.tr.300.vec" --char_embedding random --parser_path saved_models/ud_parser_qtd_sagt_full_train/ --use_unlabeled_data --model_path saved_models/ud_sequence_tagger_qtd_sagt_qtd_trde90_number_of_children_unlabeled/

###### Distance to the Root Task (DTR):
    - python examples/SequenceTagger_for_DA.py --dataset ud --src_domain qtd_sagt --tgt_domain qtd_trde90 --task distance_from_the_root --rnn_mode LSTM --num_epochs 100 --batch_size 16 --hidden_size 512 --tag_space 128 --num_layers 3 --num_filters 100 --use_char  --use_pos --char_dim 100 --pos_dim 100 --initializer xavier --opt adam --learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 --p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --unk_replace 0.5 --punct_set '.' '``'  ':' ','  --word_embedding fasttext --word_path "data/multilingual_word_embeddings/cc.tr.300.vec" --char_embedding random --parser_path saved_models/ud_parser_qtd_sagt_full_train/ --use_unlabeled_data --model_path saved_models/ud_sequence_tagger_qtd_sagt_qtd_trde90_distance_from_the_root_unlabeled/

###### Relative POS Encoding Task (RPE):
    - python examples/SequenceTagger_for_DA.py --dataset ud --src_domain qtd_sagt --tgt_domain qtd_trde90 --task relative_pos_based --rnn_mode LSTM --num_epochs 100 --batch_size 16 --hidden_size 512 --tag_space 128 --num_layers 3 --num_filters 100 --use_char  --use_pos --char_dim 100 --pos_dim 100 --initializer xavier --opt adam --learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 --p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --unk_replace 0.5 --punct_set '.' '``'  ':' ','  --word_embedding fasttext --word_path "data/multilingual_word_embeddings/cc.tr.300.vec" --char_embedding random --parser_path saved_models/ud_parser_qtd_sagt_full_train/ --use_unlabeled_data --model_path saved_models/ud_sequence_tagger_qtd_sagt_qtd_trde9_relative_pos_based_unlabeled/
    
###### Language ID of Head Task (LIH):
    - python examples/SequenceTagger_for_DA.py --dataset ud --src_domain qtd_sagt --tgt_domain qtd_trde90 --task head_lang_ids --rnn_mode LSTM --num_epochs 100 --batch_size 16 --hidden_size 512 --tag_space 128 --num_layers 3 --num_filters 100 --use_char  --use_pos --char_dim 100 --pos_dim 100 --initializer xavier --opt adam --learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 --p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --unk_replace 0.5 --punct_set '.' '``'  ':' ','  --word_embedding fasttext --word_path "data/multilingual_word_embeddings/cc.tr.300.vec" --char_embedding random --parser_path saved_models/ud_parser_qtd_sagt_full_train/ --use_unlabeled_data --model_path saved_models/ud_sequence_tagger_qtd_sagt_qtd_trde9_head_lang_ids_unlabeled/
    
###### Simplified Morphology of Head Task (SMH):
    - python examples/SequenceTagger_for_DA.py --dataset ud --src_domain qtd_sagt --tgt_domain qtd_trde90 --task head_simplified_morp_feats --rnn_mode LSTM --num_epochs 100 --batch_size 16 --hidden_size 512 --tag_space 128 --num_layers 3 --num_filters 100 --use_char  --use_pos --char_dim 100 --pos_dim 100 --initializer xavier --opt adam --learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 --p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --unk_replace 0.5 --punct_set '.' '``'  ':' ','  --word_embedding fasttext --word_path "data/multilingual_word_embeddings/cc.tr.300.vec" --char_embedding random --parser_path saved_models/ud_parser_qtd_sagt_full_train/ --use_unlabeled_data --model_path saved_models/ud_sequence_tagger_qtd_sagt_qtd_trde9_head_simplified_morp_feats_unlabeled/

###### Punctuation Count Task (PC):
    - python examples/SequenceTagger_for_DA.py --dataset ud --src_domain qtd_sagt --tgt_domain qtd_trde90 --task count_punct --rnn_mode LSTM --num_epochs 100 --batch_size 16 --hidden_size 512 --tag_space 128 --num_layers 3 --num_filters 100 --use_char  --use_pos --char_dim 100 --pos_dim 100 --initializer xavier --opt adam --learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 --p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --unk_replace 0.5 --punct_set '.' '``'  ':' ','  --word_embedding fasttext --word_path "data/multilingual_word_embeddings/cc.tr.300.vec" --char_embedding random --parser_path saved_models/ud_parser_qtd_sagt_full_train/ --use_unlabeled_data --model_path saved_models/ud_sequence_tagger_qtd_sagt_qtd_trde9_count_punct_unlabeled/
    
     

##### 1.4- Train the final model:

Now that we trained all of the auxiliary task enhancement models, we can train the final model. Let's say we want to train the best performing model in the paper for Turkish-German SAGT Treebank: +RPE,+LIH,+SMH which stands for the ensemble of RPE, LIH, and SMH tasks.

    - python examples/GraphParser_for_DA.py --dataset ud --src_domain qtd_sagt --tgt_domain qtd_trde90 --rnn_mode LSTM --num_epochs 150 --batch_size 16 --hidden_size 512 --arc_space 512 --arc_tag_space 128 --num_layers 3 --num_filters 100 --use_char --use_pos  --word_dim 300 --char_dim 100 --pos_dim 100 --initializer xavier --opt adam --learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 --p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --arc_decode mst --unk_replace 0.5 --punct_set '.' '``'  ':' ','  --word_embedding fasttext --word_path "data/multilingual_word_embeddings/cc.tr.300.vec" --char_embedding random --gating --num_gates 4 --load_sequence_taggers_paths saved_models/ud_sequence_tagger_qtd_sagt_qtd_trde90_relative_pos_based_unlabeled/src_domain_qtd_sagt_tgt_domain_qtd_trde90.pt saved_models/ud_sequence_tagger_qtd_sagt_qtd_trde90_head_of_lang_ids_unlabeled/src_domain_qtd_sagt_tgt_domain_qtd_trde90.pt saved_models/ud_sequence_tagger_qtd_sagt_qtd_trde90_head_simplified_morp_feats_unlabeled/src_domain_qtd_sagt_tgt_domain_qtd_trde90.pt --model_path saved_models/ud_parser_qtd_sagt_qtd_trde90_ensemble_gating_RPE_LIH_SMH/

----------------------------------------------------------------

If you want to join only one seq_labeler model (e.g., only +NOC model), set --num_gates to 2 and provide only the trained model of that seq_labeler model.



****************************************************************
----------------------------------------------------------------

## ## How-To-Run the XLM-R-based Parser:

### Requirements

Create a conda environment using the environment.yml file:

   - conda env create -f  XLM-R-based/auxiliary-task-train/steps_parser/environment.yml 
  
Activate the environment:

   - conda activate ss_cs_depparse
    
    
### Pretrained Language Model

Download XLM-R base model from [Hugging Face](https://huggingface.co/xlm-roberta-base/tree/main) and locate it under 
**XLM-R-based/dcst-parser-train/pretrained_model/**.


### Datasets

- For labeled data, we use the QTD_SAGT dataset from [Universal Dependencies](https://github.com/UniversalDependencies): 
    - Download QTD_SAGT treebank and locate it under **LSTM-based/DCST/data/datasets/**
- For unlabeled data, we use ["TuGeBiC"](https://github.com/ozlemcek/TuGeBiC). 
    - Download the corpus, join all conll-u files and divide them to train and dev files. Name the training as "qtd_trde90-ud-train_autoparsed.conllu" and dev as "qtd_trde90-ud-dev_autoparsed.conllu". Locate these files under **XLM-R-based/auxiliary-task-train/preprocessed_unlabeled_data/**
    
#### Preprocess Unlabeled Data

Navigate to **XLM-R-based/auxiliary-task-train/preprocessed_unlabeled_data/**

Run the corresponding Python script for the auxiliary task you want to use. E.g., for the LIH task:

    - python dcst_langid_of_head.py qtd_trde90-ud-train_autoparsed.py qtd_trde90-ud-train_autoparsed_lih.py
    - python dcst_langid_of_head.py qtd_trde90-ud-dev_autoparsed.py qtd_trde90-ud-dev_autoparsed_lih.py
    
### Trained Models

Download the trained models from the [Trained_Models_XLM-R folder](https://drive.google.com/drive/folders/12F4ieakslvFZtOAj4JOqRX3NOTLPLICs?usp=sharing). Locate parser_models under **XLM-R-based/dcst-parser-train/trained_models/** and auxiliary_task_models under **XLM-R-based/auxiliary-task-train/trained_models/**

 --------------
### Use the Trained Model to Parse QTD_SAGT:

Let's say we want to use the +LIH model for Tr-De CS pair (QTD_SAGT).

    - cd XLM-R-based/dcst-parser-train/
   
    - python src/train.py ../deps_lih_qtd.json
    

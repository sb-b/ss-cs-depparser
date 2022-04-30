# Semi-Supervised CS Dependency Parser

This page includes source codes and trained models of the semi-supervised deep dependency parser described in this paper. The parser employs a semi-supervised learning approach ["DCST"](https://rotmanguy.github.io/publications/2019-10-01-deep-contextualized-self-learning) and utilizes auxiliary tasks for dependency parsing of code-switched (CS) language pairs. There are two versions of the parsing model, one is LSTM-based and the other is XLM-R-based. The following sections explain how to run these models.


### Requirements

Run the following:

    - pip install -r requirements.txt
    
### Datasets

* Navigate to the DCST folder

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

## How-To-Run the LSTM-based Parser

Let's say we want to train the LSTM-based model with auxiliary task enhancements for the Turkish-German SAGT Treebank (qtd_sagt). As the unlabeled data, we use ["TuGeBiC"](https://github.com/ozlemcek/TuGeBiC) (qtd_trde90).

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

### 2- XLM-R-based DCST (using STEPS):

#### Prerequisites:

This model does not run on steppenweihe. So, we run the models on grauweihe.

First, activate the conda environment:

##### On Grauweihe server:

    - conda activate depparse_grau

--------------
#### Example runs shown for KPV_ZYRIAN:


- used datasets: 
    - labeled_data: /mount/projekte/codeswitch/betul/DCST/data/datasets/UD_Komi-Zyrian
    - unlabeled_data /mount/projekte/codeswitch/betul/DCST/data/datasets/UD_Komi-Social

Let's say we want to train the +SMH model for Kpv-Ru CS pair.

##### 2.1- Train the baseline parser:

    - cd /mount/projekte/codeswitch/betul/steps_parser_2/steps_parser/

    - python src/train.py ../deps_only_kpv-2-8.json

##### 2.2- Parse the unlabeled data:

Step 1- Open 'train.py' in /mount/projekte/codeswitch/betul/steps_parser_2/steps_parser/src/

Step 2- Comment out line 33 like this:

    - # trainer.train()

Step 3- Comment out line 48 like this:

    - # checkpoint_path = Path(trainer.checkpoint_dir) / "model_best.pth"

Step 4- Delete '#' char at the beginning of line 47 and provide the trained baseline model path. e.g.:

    - checkpoint_path = "/mount/projekte18/codeswitch/betul/steps_parser_2/trained_models/KPV_Zyrian-2.8_baseline/baseline_only_deps_batchsize_16_es_200/1130_142937/model_best.pth"


Step 5- Open 'deps_only_kpv-2-8.json' in /mount/projekte/codeswitch/betul/steps_parser_2/

Step 6- Provide the path of the unlabeled data in line 94. E.g.:

    - "test": "/mount/projekte18/codeswitch/betul/DCST/data/datasets/UD_Komi-Social/kpv_social-ud-train.conllu"

- The newly created file 'test-parsed.conllu' file under /mount/projekte/codeswitch/betul/steps_parser_2/steps_parser/ is the automatically parsed version of 'kpv_social-ud-train.conllu'

- Do this again for dev and test parts of the unlabeled data. (The treebank in /mount/projekte18/codeswitch/betul/DCST/data/datasets/UD_Komi-Social/ is already the automatically parsed version. I just wanted to show this process for a new data).

Step 7- Undo steps 2, 3, and 4.

##### 2.3- Train sequence labelers:

###### Simplified Morphology of Head Task:

Step 1- Navigate to /mount/projekte/codeswitch/betul/all_purpose_scripts/steps_gating/

    - python dcst_simplified_morp_features_of_head.py /mount/projekte18/codeswitch/betul/DCST/data/datasets/UD_Komi-Social/kpv_social-ud-train.conllu kpv_social-ud-train-parsedbysteps-smh.conllu

    - python dcst_simplified_morp_features_of_head.py /mount/projekte18/codeswitch/betul/DCST/data/datasets/UD_Komi-Social/kpv_social-ud-dev.conllu kpv_social-ud-dev-parsedbysteps-smh.conllu

    - python dcst_simplified_morp_features_of_head.py /mount/projekte18/codeswitch/betul/DCST/data/datasets/UD_Komi-Social/kpv_social-ud-test.conllu kpv_social-ud-test-parsedbysteps-smh.conllu

(These preprocessed files are already exist in /mount/projekte/codeswitch/betul/all_purpose_scripts/steps_gating/)

*For experimenting with other tasks:*

  - Use:
  
     - dcst_num_of_children.py for (+NOC),
     - dcst_distance_to_root.py for (+DTR),
     - dcst_relative_pos_tag.py for (+RPE),
     - dcst_simplified_head_lang_id.py for (+HLI),
     - dcst_count_punct.py for (+PC),
     
    instead of dcst_simplified_morp_features_of_head.py

Step 2- Open 'init_config.py' in /mount/projekte/codeswitch/betul/steps_parser_2/steps_parser/src/

Step 3- Replace line 126 

    - post_processors =  self._init_post_processors(model_args["post_processors"], model_outputs))
    
    
   with:
    
    - post_processors = [] 

Step 4- In /mount/projekte/codeswitch/betul/steps_parser_2/steps_parser/ :

Run

    - python src/train.py ../dcst_simplified_morp_feats_of_head_social_parsedbykpv.json
    
    
*For experimenting with other tasks:*

  - Use:
  
     - dcst_number_of_children_social_parsedbykpv.json for (+NOC),
     - dcst_distance_to_root_social_parsedbykpv.json for (+DTR),
     - dcst_relative_pos_tag_social_parsedbykpv.json for (+RPE),
     - dcst_head_lang_id_social_parsedbykpv.json for (+HLI),
     - dcst_punct_count_social_parsedbykpv.json for (+PC),
     
    instead of dcst_simplified_morp_feats_of_head_social_parsedbykpv.json
    
Step 5- Undo step 3.

##### 2.4- Train the final ensemble model (+SMH):

Step 1- Open 'train.py' in /mount/projekte/codeswitch/betul/steps_gating_oneseq/steps_parser/src/

Step 2- Uncomment the corresponding seq_trainer.resume_checkpoint command. E.g., it is line 101 for SMH task of Komi_Zyrian.

Step 3- In /mount/projekte/codeswitch/betul/steps_gating_oneseq/steps_parser/ :

Run

    - python src/train.py ../deps_smh_kpv.json
    
------------------------------




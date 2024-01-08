# DLCykParser

## Introduction

The CYK algorithm can decide whether a given string of any length is in a given context-free grammar or not. However, its use cases are limited, because the production grammar must be available as an input parameter, hence the algorithm being inapplicable in scenarios where the context-free grammar is unknown.
In this work, we build a neuralized version of the classical CYK algorithm, such that it can learn and generalize to more softly defined predictions. To that end, we build a neuralized version of the classical CYK Parser, which can learn purely by examples and can be trained end-to-end in a supervised setting.

## Usage

All of the grammars, datasets, and experiment results are in the folder "data"

- To get the results of NCYK, run: `python neuro_cyk_parser_v2.py <path/to/dataset/folder> <num_of_rules> <num_of_epochs>(optional)` E.g. To run the the random grammar 3 small dataset using 6 rules for 10 epochs, run: `python neuro_cyk_parser_v2.py data/random/grammar3/small_ds 6 10`
- To get the results of our baselines, run: `python lstm.py` and change pathname in file or `python transformer.py <path/to/dataset/folder>`
- To generate random grammars, run: `python generate_random_grammar.py --valid_symbols <valid_symbols> --max_rules <max_rules> --max_different_rules <max_different_rules> --loc <path/to/output> --seed <seed>(optional)`
- To generate sequences for a random grammar, run: `python seq_generator.py <min_train_length> <max_train_length> <min_test_length> <max_test_length> <num_train_samples> <num_id_samples> <num_ood_samples> <loc_grammar> <loc_dataset_dir>`

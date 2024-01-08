# DLCykParser
## Usage
- To get the results of NCYK, run: ```python neuro_cyk_parser_v2.py <path/to/dataset/folder> <num_of_rules> <num_of_epochs>(optional)```
- To get the results of our baselines, run: ```python lstm.py <path/to/dataset/folder>``` or ```python transformer.py <path/to/dataset/folder>```
- To generate random grammars, run: ```python generate_random_grammar.py --valid_symbols <valid_symbols> --max_rules <max_rules> --max_different_rules <max_different_rules> --loc <path/to/output> --seed <seed>(optional)```
- To generate sequences for a random grammar, run: ```python seq_generator.py <min_train_length> <max_train_length> <min_test_length> <max_test_length> <num_train_samples> <num_id_samples> <num_ood_samples> <loc_grammar> <loc_dataset_dir>```

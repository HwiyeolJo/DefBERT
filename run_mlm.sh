python MLM.py \
	--model_name_or_path bert-base-uncased \
	--per_device_train_batch_size 128 \
	--per_device_eval_batch_size 64 \
	--train_file ./data/DefinitionDataset.txt \
	--line_by_line \
	--output_dir /ckpt/def-bert/save/mlm/bert \
	--overwrite_output_dir \
	--do_train \
	--learning_rate 1e-5 \
	--num_train_epochs 10 \
	--save_strategy epoch
# 	--model_name_or_path sentence-transformers/bert-base-nli-stsb-mean-tokens \
# 	--model_name_or_path bert-base-uncased \

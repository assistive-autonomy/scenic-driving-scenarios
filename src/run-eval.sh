#! /bin/bash

for model_name in "codellama/CodeLlama-7b-Instruct-hf" "mistralai/Mistral-7B-Instruct-v0.2" "google/gemma-7b-it" "meta-llama/Meta-Llama-3-8B-Instruct"; do
	for random_examples in False True; do
		for retrieval_top_k in 1 2 3; do
			python exp.py model.model_name="$model_name" model.random_examples=$random_examples model.retrieval_top_k=$retrieval_top_k model.exceptions.do_feeding=False
			for error_feeding_max_trials in 2 3 4 5; do
				python exp.py model.model_name="$model_name" model.random_examples=$random_examples model.retrieval_top_k=$retrieval_top_k model.exceptions.do_feeding=True model.exceptions.max_trials=$error_feeding_max_trials
			done
		done
	done
done



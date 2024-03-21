import os

from datasets import load_dataset
from transformers import (DataCollatorForSeq2Seq,
						  Seq2SeqTrainer,
						  Seq2SeqTrainingArguments,
						  AutoTokenizer,
						  AutoModelForSeq2SeqLM)

from peft import get_peft_model, LoraConfig, TaskType
import evaluate

"""set WANDB logging"""
os.environ["WANDB_PROJECT"] = "cdsg-experiments"
os.environ["WANDB_LOG_MODEL"] = "true"

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", padding=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSeq2SeqLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
peft_model = get_peft_model(model, config)

def description2program(examples):
	"""task of description to program generation."""
    
	inputs = tokenizer(examples["description"], return_tensors="pt", padding=True, truncation=True)
	labels = tokenizer(examples["program"], return_tensors="pt", padding=True, truncation=True)
	return{
		"input_ids": inputs["input_ids"],
		"attention_mask": inputs["attention_mask"],
		"labels": labels["input_ids"],
	}

dataset = load_dataset("ipab-rad/driving_scenarios", trust_remote_code=True)
d2p_dataset = dataset.map(description2program, batched=True)

# load metrics
blue_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
exact_match_metric = evaluate.load("exact_match")
perprexity_metric = evaluate.load("perplexity", module_type="metric")

def compute_metrics(eval_pred):
	"""compute metrics using evaluate: BLUE, ROUGE, F1"""
	predictions, labels = eval_pred
	decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
	decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

	bleu_score = blue_metric.compute(predictions=decoded_preds, references=[decoded_labels])
	rouge_score = rouge_metric.compute(predictions=decoded_preds, references=[decoded_labels])
	exact_match_metric = exact_match_metric.compute(predictions=decoded_preds, references=[decoded_labels])
	perprexity_score = perprexity_metric.compute(predictions=decoded_preds, references=[decoded_labels], model_id="gpt2")

	return {"bleu": bleu_score,
		 "rouge": rouge_score,
		 "exact_match": exact_match_metric,
		 "perprexity": perprexity_score,
		 }



training_args = Seq2SeqTrainingArguments(
    output_dir="./models/lora",
    learning_rate=1e-3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="wandb",
    run_name="lora",
    logging_steps=1,
)

# Define data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
)

trainer = Seq2SeqTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=d2p_dataset["train"],
    eval_dataset=d2p_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

peft_model.save_pretrained("./models/lora")



import datasets
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
import evaluate
import numpy as np
from transformers import EarlyStoppingCallback

rouge = evaluate.load("rouge")
xsum = datasets.load_dataset("xsum")
access_token = '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'

model_name = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
  return tokenizer([" ".join(x) for x in examples['summary']])

tokenized_xsum = xsum.map(
  preprocess_function,
  batched=True,
  num_proc=4,
  remove_columns=xsum["train"].column_names,
)

block_size = 128


def group_texts(examples):
  # Concatenate all texts.
  concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
  total_length = len(concatenated_examples[list(examples.keys())[0]])
  # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
  # customize this part to your needs.
  if total_length >= block_size:
      total_length = (total_length // block_size) * block_size
  # Split by chunks of block_size.
  result = {
      k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
      for k, t in concatenated_examples.items()
  }
  result["labels"] = result["input_ids"].copy()
  return result

lm_dataset = tokenized_xsum.map(group_texts, batched=True, num_proc=4)

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

model = AutoModelForCausalLM.from_pretrained(model_name)

training_args = TrainingArguments(
    output_dir="falcon_xsum_finetuned_train",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    push_to_hub=True,
    hub_token = access_token,
    hub_strategy="all_checkpoints",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    do_train=True,
    do_eval=True,
    num_train_epochs=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["validation"],
    data_collator=data_collator,
    # compute_metrics = compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=5)],
)

trainer.train()
trainer.push_to_hub()
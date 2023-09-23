import datasets
from transformers import T5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
import torch
import evaluate
import numpy as np
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback
import pickle

xsum = datasets.load_dataset("xsum")

model_name = "t5-base"
device = "cuda" if torch.cuda.is_available() else "cpu"
t5_tokenizer = AutoTokenizer.from_pretrained(model_name)
data_collator = DataCollatorForSeq2Seq(tokenizer=t5_tokenizer, model=model_name)
t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

prefix = "summarize: "

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples['document']]
    model_inputs = t5_tokenizer(inputs, max_length=1024, truncation=True)

    labels = t5_tokenizer(text_target=examples['summary'], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
  tokenizer = t5_tokenizer
  predictions, labels = eval_pred
  decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
  labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
  decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

  result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

  prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
  result["gen_len"] = np.mean(prediction_lens)

  return {k: round(v, 4) for k, v in result.items()}

tokenized_xsum = xsum.map(preprocess_function, batched=True)

access_token = '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'

training_args = Seq2SeqTrainingArguments(
    output_dir="xsum_finetuned_on_train",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    do_train=True,
    do_eval=True,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    # save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    load_best_model_at_end=True,
    metric_for_best_model="rouge1",
    fp16=True,
    push_to_hub=True,
    hub_token = access_token,
    hub_strategy="all_checkpoints",
)


trainer = Seq2SeqTrainer(
    model=t5_model,
    args=training_args,
    train_dataset=tokenized_xsum["train"],
    eval_dataset=tokenized_xsum["validation"],
    tokenizer=t5_tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=5)],
)


trainer.train()
trainer.push_to_hub()

model_name = "mpalaval/xsum_finetuned_on_train"
# summarizer = pipeline("summarization", model="mpalaval/exp2_xsum_model")
summarizer = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
t5_tokenizer = AutoTokenizer.from_pretrained(model_name)
# device = "cuda" if torch.cuda.is_available() else "cpu"

preds, gts = [], []
counter = 0
prefix = "summarize: "
for idx in range(xsum['test'].num_rows):
  # summary = summarizer(prefix + xsum['test'][idx]['document'])
  batch = t5_tokenizer(prefix + xsum['test'][idx]['document'], truncation=True, padding="longest", return_tensors="pt").to(device)
  translated = summarizer.generate(**batch)
  tgt_text = t5_tokenizer.batch_decode(translated, skip_special_tokens=True)

  preds.append(tgt_text[0])
  gts.append(xsum['test'][idx]['summary'])

with open('predictions_exp2.pkl', 'wb') as f:
  pickle.dump(preds, f)
with open('gts_exp2.pkl', 'wb') as f:
  pickle.dump(gts, f)

rouge = evaluate.load("rouge")
result = rouge.compute(predictions=preds, references=gts, use_stemmer=True)
print(result)

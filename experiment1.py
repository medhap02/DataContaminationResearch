import datasets
from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch
import pickle
import evaluate


xsum = datasets.load_dataset("xsum")

model_name = "t5-base"
device = "cuda" if torch.cuda.is_available() else "cpu"
t5_tokenizer = AutoTokenizer.from_pretrained(model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

preds, gts = [], []
counter = 0
prefix = "summarize: "
for idx in range(xsum['test'].num_rows):
  batch = t5_tokenizer(prefix + xsum['test'][idx]['document'], truncation=True, padding="longest", return_tensors="pt").to(device)
  translated = t5_model.generate(**batch)
  tgt_text = t5_tokenizer.batch_decode(translated, skip_special_tokens=True)
  # data_idx = dataset['test'][idx]['id']
  preds.append(tgt_text[0])
  gts.append(xsum['test'][idx]['summary'])
  # counter += 1
  # if counter % 10 == 0:
  #   print('done with test', counter)

with open('predictions_exp1', 'wb') as f:
  pickle.dump(preds, f)
with open('gts_exp1', 'wb') as f:
  pickle.dump(gts, f)

rouge = evaluate.load("rouge")
result = rouge.compute(predictions=preds, references=gts, use_stemmer=True)
print(result)
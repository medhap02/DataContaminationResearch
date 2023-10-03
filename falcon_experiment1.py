import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import pickle
import evaluate

xsum = datasets.load_dataset("xsum")

model_name = "tiiuae/falcon-rw-1b"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/ocean/projects/cis230007p/palavall/DataContaminationResearch/cache/")
pipeline = transformers.pipeline(
    "text-generation",
    model=model_name,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

preds, gts = [], []
prefix = "summarize: "

for idx in range(xsum['test'].num_rows):
  sequences = pipeline(
    xsum['test'][idx]['document'],
    max_new_tokens = 100,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
  )
  output = sequences[0]['generated_text']
  preds.append(output)
  gts.append(xsum['test'][idx]['summary'])

with open('falcon_preds_exp1', 'wb') as f:
  pickle.dump(preds, f)
with open('falcon_gts_exp1', 'wb') as f:
  pickle.dump(gts, f)

rouge = evaluate.load("rouge")
result = rouge.compute(predictions=preds, references=gts, use_stemmer=True)
print(result)

import torch
import gc
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoConfig
from datasets import load_dataset
import evaluate

# model name
model_name = "gpt2-xl"

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
# load  model
config = AutoConfig.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,config=config)

# load dataset
dataset = load_dataset("squad")
train_ds = dataset['train'].shuffle(seed=42)
vali_ds = dataset['validation'].shuffle(seed=42)
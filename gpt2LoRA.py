import torch
import gc
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoConfig
from datasets import load_dataset
import evaluate

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
# load  model
config = AutoConfig.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained("gpt2",config=config)

# load dataset
dataset = load_dataset("squad")
vali_ds = dataset['train'].select(range(5))
spilt_ds = dataset['train'].train_test_split(test_size=0.1)
train_ds = spilt_ds['train'].shuffle(seed=42).select(range(2000))
eval_ds = spilt_ds['test'].shuffle(seed=42).select(range(200))
# clear original dataset
del dataset
del spilt_ds
gc.collect()

# define format prompt function
def format_prompts(examples):
    prompts = [f"Context: {c}\nQuestion: {q}\n" for q, c in zip(examples['question'], examples['context'])]
    return tokenizer(prompts, padding="max_length", truncation=True, max_length=256, return_tensors='pt')


from peft import LoraConfig, TaskType, get_peft_model
config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    fan_in_fan_out=True,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    )

lora_model = get_peft_model(model,config)
lora_model.print_trainable_parameters()

#define preprocess function
def preprocess_function(examples):
    inputs = [f"Context: {c}\nQuestion: {q}\nAnswer: " for q, c in zip(examples['question'], examples['context'])]
    model_inputs = tokenizer(inputs,padding="max_length", truncation=True, max_length=256,return_tensors='pt')
    
    targets = [','.join(a['text']) if len(a['text']) > 0 else '' for a in examples['answers']]
    labels = tokenizer(targets,padding="max_length", truncation=True, max_length=256, return_tensors='pt')
    model_inputs["labels"] = labels['input_ids']
    model_inputs["labels_mask"] = labels['attention_mask']
    return model_inputs

tok_train_ds = train_ds.map(preprocess_function, batched=True)
tok_train_ds.set_format(type="torch", columns=["input_ids", "attention_mask","labels","labels_mask"])
print(tok_train_ds)
tok_eval_ds = eval_ds.map(preprocess_function, batched=True)
tok_eval_ds.set_format(type="torch", columns=["input_ids", "attention_mask","labels","labels_mask"])

# prepare for training
from transformers import TrainingArguments,Trainer
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    output_dir="./results",
    learning_rate=2e-4,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    num_train_epochs=3,
    )
trainer = Trainer(model=lora_model,
                  args=training_args,
                  tokenizer=tokenizer,
                  train_dataset=tok_train_ds,
                  eval_dataset=tok_eval_ds,
                    )
# train
trainer.train()
# eval
evaluation_result = trainer.evaluate()
print(evaluation_result)
#save
lora_model.save_pretrained("gpt2-lora")
# load fine-tuning model
from peft import AutoPeftModelForCausalLM
ft_model = AutoPeftModelForCausalLM.from_pretrained("gpt2-lora")



inputs = format_prompts(vali_ds)
outputs = ft_model.generate(**inputs, max_new_tokens=32,do_sample=True, top_k=50, top_p=0.95, temperature=0.7,num_return_sequences=1)

# generate anwsers
generated_answers = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# display result
for input, output, ans in zip(inputs['input_ids'], outputs,generated_answers):
    print("-------compare--------")
    print(f"input:\n{tokenizer.decode(input,skip_special_tokens=True)}")
    print(f"output:\n{tokenizer.decode(output,skip_special_tokens=True)}")
    print(f"answer:\n{ans}")

# evaluate based on squad metric
ft_squad_metric = evaluate.load("./metrics/squad")

# prepare prediciotns
predictions = [{'id': example['id'],
                'prediction_text': answer} for example, answer in zip(vali_ds, generated_answers)]

# prepare references
references = [{'id': example['id'], 
               'answers': example['answers']} for example in vali_ds]
# compute and print result
ft_results = ft_squad_metric.compute(predictions=predictions, references=references)
print(ft_results)
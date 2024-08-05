import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

model_name = "gpt2-xl" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
ds = load_dataset("squad")

# Freeze model parameters
for param in model.parameters():
    param.requires_grad = False

# Define soft prompts (θp for task-specific tuning, θs for self-evaluation)
### why 5? - 5-shot prompt?
soft_prompt_task = torch.nn.Parameter(torch.randn(5, model.config.n_embd))
soft_prompt_eval = torch.nn.Parameter(torch.randn(5, model.config.n_embd))

# Define optimizer for soft prompts
optimizer = torch.optim.Adam([soft_prompt_task, soft_prompt_eval], lr=1e-5)

# Fine-tuning process
def fine_tune_step(input_ids, labels):
    # Add soft prompt to input embeddings
    input_embeddings = model.transformer.wte(input_ids)
    input_embeddings = torch.cat([soft_prompt_task.unsqueeze(0), input_embeddings], dim=1)
    
    # Forward pass
    outputs = model(inputs_embeds=input_embeddings, labels=labels)
    loss = outputs.loss
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()

def self_evaluate(input_ids, response_ids, soft_prompt_task, soft_prompt_eval):
    """
    生成不同的回答并进行自我评估。
    
    Args:
        input_ids (torch.Tensor): 查询的输入token IDs。
        response_ids (torch.Tensor): 模型生成的响应token IDs。
        soft_prompt_task (torch.Tensor): 任务特定的软提示。
        soft_prompt_eval (torch.Tensor): 自我评估的软提示。
    
    Returns:
        str: 任务回答。
        str: 自评估回答。
        float: 自我评估得分。
    """
    # 将软提示和输入embedding拼接
    input_embeddings_query = model.transformer.wte(input_ids)
    input_embeddings_response = model.transformer.wte(response_ids)
    input_embeddings_task = torch.cat([soft_prompt_task.unsqueeze(0), input_embeddings_query, input_embeddings_response], dim=1)
    input_embeddings_eval = torch.cat([soft_prompt_eval.unsqueeze(0), input_embeddings_query, input_embeddings_response], dim=1)
    
    # 使用任务特定的软提示生成答案
    task_outputs = model.generate(inputs_embeds=input_embeddings_task, max_length=50)
    eval_outputs = model.generate(inputs_embeds=input_embeddings_eval, max_length=50)
    
    # 解码答案
    task_answer = tokenizer.decode(task_outputs[0], skip_special_tokens=True)
    eval_answer = tokenizer.decode(eval_outputs[0], skip_special_tokens=True)
    
    # 计算自我评估得分
    score = compute_self_evaluation_score(task_answer, eval_answer)
    return task_answer, eval_answer, score

# process
# 1. task specific fine-tuning
## 1) freeze theta
for param in model.parameters():
    param.requires_grad = False
## 2) initialize theta_p
theta_p = torch.nn.Parameter(torch.randn(5, model.config.n_embd))
## 3) train theta_p
num_epochs = 1e3
# given: dataset, num_train_p, learning_rate
def train_p(train_ds, num_epochs, lr):
    optimizer = torch.optim.Adam(theta_p, lr=lr)
    loss_func = torch.nn.CrossEntropyLoss()
    ds_iter = iter(train_ds)
    for _ in range(num_epochs):
        inputs = model.transformer.wte(ds_iter['context'] + ds_iter['question'])
        inputs = torch.cat([theta_p.unsqueeze(0), inputs], dim=1)
        next(ds_iter)

        outputs = model(inputs) 

        labels = model.transformer.wte(ds_iter['answers']['text'])
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
# 2. answer sampling


# 3. self-evaluation



# Example fine-tuning
input_text = "Which vitamin assists in blood clotting?"
inputs = tokenizer(input_text, return_tensors="pt")
input_ids = inputs.input_ids
labels = tokenizer("Vitamin K", return_tensors="pt").input_ids  # Example label
loss = fine_tune_step(input_ids, labels)
print(f"Fine-tuning loss: {loss}")

task_answer, eval_answer, score = self_evaluate(input_ids, soft_prompt_task, soft_prompt_eval)
print(f"Task Answer: {task_answer}")
print(f"Eval Answer: {eval_answer}")
print(f"Self-Evaluation Score: {score}")

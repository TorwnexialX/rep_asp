import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2-xl" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

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

# Generate different answers and perform self-evaluation
def self_evaluate(input_ids, soft_prompt_task, soft_prompt_eval):
    # Add soft prompts to input embeddings
    input_embeddings = model.transformer.wte(input_ids)
    input_embeddings_task = torch.cat([soft_prompt_task.unsqueeze(0), input_embeddings], dim=1)
    input_embeddings_eval = torch.cat([soft_prompt_eval.unsqueeze(0), input_embeddings], dim=1)
    
    # Generate answers using task-specific prompt
    task_outputs = model.generate(inputs_embeds=input_embeddings_task, max_length=50)
    eval_outputs = model.generate(inputs_embeds=input_embeddings_eval, max_length=50)
    
    # Decode answers
    task_answer = tokenizer.decode(task_outputs[0], skip_special_tokens=True)
    eval_answer = tokenizer.decode(eval_outputs[0], skip_special_tokens=True)
    
    # Compute self-evaluation score
    # Note: Implement your own scoring function based on the document details
    # For example, comparing with a reference answer or using a confidence score
    score = compute_self_evaluation_score(task_answer, eval_answer)
    return task_answer, eval_answer, score

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

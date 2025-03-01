
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import tensorflow as tf

dataset = load_dataset("conll2003")

def preprocess_function(examples):
    tokenized_inputs = tokenizer(examples['sentence'], truncation=True, padding=True)
    return tokenized_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
)


trainer.train()

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# Load the prepared dataset
dataset = load_dataset('text', data_files={'train': '/kaggle/input/dataset-dd/DD.txt'})

# Initialize the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Set the EOS token as the padding token
tokenizer.pad_token = tokenizer.eos_token


# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

 # Set up the data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Define training arguments
training_args = TrainingArguments(
output_dir='./results',
overwrite_output_dir=True,
num_train_epochs=25,  # Increased from 5 to 25 epochs to overfit
per_device_train_batch_size=8,  # Batch size per device during training
save_steps=10_000,  # Save the model every 10,000 steps
save_total_limit=1,  # Limit the total amount of checkpoints
logging_dir='./logs',  # Directory for storing logs
logging_steps=200,  # Log every 200 steps
learning_rate=5e-5,  # Learning rate
weight_decay=0.0,  # Set to 0 to reduce regularization and promote overfitting
evaluation_strategy="no",
report_to="none"  # Disable W&B reporting
)

    # Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

    # Start fine-tuning
trainer.train()

    # Save the fine-tuned model
model.save_pretrained('MMK1')
tokenizer.save_pretrained('MMK1')

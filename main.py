import argparse
from functools import partial

from datasets import load_dataset
import optuna
from peft import get_peft_model, LoraConfig
import torch
import torch.nn as nn
import transformers


class LoRALayer(nn.Module):
    # Custom implementation of LoRA layer
    def __init__(self, base_layer, rank, alpha):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.input_features = base_layer.in_features
        self.output_features = base_layer.out_features

        # Low-rank matrices corresponding to A and B in orig. paper
        self.lora_A = nn.Linear(self.input_features, rank, bias=False)
        self.lora_B = nn.Parameter(torch.zeros(rank, self.output_features))
        self.lora_dropout = nn.Dropout(p=0.05)

    def forward(self, x):
        lora_A_output = self.lora_A(x)
        lora_B_output = self.lora_dropout(lora_A_output) @ (
            self.lora_B * (self.rank / self.alpha)
        )
        
        return self.base_layer(x) + lora_B_output


def apply_lora(model, rank, alpha=16):
    """Apply LoRA to the model

    Args:
        model (Model): The model to apply LoRA to
        rank (int): The rank of the low-rank approximation
    """
    for i, layer in enumerate(model.model.layers):
        # Apply LoRA to attention layers
        for proj in ['q_proj', 'k_proj', 'v_proj']:
            base_layer = getattr(layer.self_attn, proj)
            lora_layer = LoRALayer(base_layer, rank=rank, alpha=alpha)
            setattr(layer.self_attn, proj, lora_layer)

        # Apply LoRA to MLP layers
        for fc in ['fc1', 'fc2']:
            base_layer = getattr(layer.mlp, fc)
            lora_layer = LoRALayer(base_layer, rank=rank, alpha=alpha)
            setattr(layer.mlp, fc, lora_layer)
            

def load_peft_lora(model, rank=16, alpha=16):
    """Load LoRA from PEFT library

    Args:
        model (Model): The model to load LoRA to
        rank (int, optional): The rank of the low-rank approximation
        alpha (int, optional): The alpha parameter for LoRA

    Returns:
        Model: The model with LoRA loaded from PEFT library
    """
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj", "k_proj", "fc1", "fc2"],
        task_type="CAUSAL_LM"
    )
    model_peft = get_peft_model(model, config)
    model_peft.gradient_checkpointing = True
    
    return model_peft


def load_custom_lora(model, rank=16, alpha=16):
    """Load custom LoRA implementation

    Args:
        model (Model): The model to load LoRA to
        rank (int, optional): The rank of the low-rank approximation

    Returns:
        Model: The model with custom LoRA implementation
    """
    apply_lora(model, rank, alpha=alpha)
    
    return model


def load_pretrained_model(model_name="microsoft/phi-1_5"):
    """Load the model and tokenizer

    Args:
        model_name (str, optional): The model name to load.

    Returns:
        Model: The model
        Tokenizer: The tokenizer
    """
    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model and freeze the pretrained weights
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    for param in model.parameters():
        param.requires_grad = False

    return model, tokenizer


def tokenize_line(line, tokenizer, max_length=250):
    inputs = tokenizer(line["text"], padding='max_length', truncation=True,
                       max_length=max_length, return_tensors="pt")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(line['output'], padding='max_length',
                           truncation=True, max_length=max_length,
                           return_tensors="pt").input_ids
    inputs['labels'] = labels

    return inputs


def train_model(model, learning_rate, tokenizer, train_tokenized, 
                test_tokenized, output_dir='results'):
    """Train the model

    Args:
        model (Model): The model to train
        learning_rate (float): The learning rate
        tokenizer (Tokenizer): The tokenizer
        train_tokenized (Dataset): The train dataset
        test_tokenized (Dataset): The test dataset

    Returns:
        Model: The trained model
    """
    # Fine-tune the model
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=8,
            warmup_steps=100,
            num_train_epochs=3,
            logging_steps=50,
            learning_rate=learning_rate,
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy='epoch',
            do_eval=True,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        )
    )
    trainer.train()
    
    # Save the model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return trainer


def tune_hyperparameters(model_name, lora_im, train_tokenized, 
                         test_tokenized, n_trials=10):
    """Tune hyperparameters for LoRA fine-tuning using BayesOpt

    Args:
        model_name (str): The model name
        lora_im (str): The LoRA implementation
        tokenizer (Tokenizer): The tokenizer
        train_tokenized (Dataset): The train dataset
        test_tokenized (Dataset): The test dataset

    Returns:
        dict: The best hyperparameters
    """
    def _objective(trial):
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        rank = trial.suggest_int("r", 8, 32)
        alpha = trial.suggest_int("alpha", 8, 32)
        
        # Load model with hyperparams
        model, tokenizer = load_pretrained_model(model_name)
        if lora_im == "peft":
            model = load_peft_lora(model, rank=rank, alpha=alpha)
        elif lora_im == "custom":
            model = load_custom_lora(model, rank=rank, alpha=alpha)
        else:
            raise ValueError(
                "Invalid LoRA implementation. Choose 'peft' or 'custom'."
            )
        
        # Train model
        trainer = train_model(model, lr, tokenizer, 
                              train_tokenized, test_tokenized)
         
        # Evaluate model
        eval_metrics = trainer.evaluate()
        
        return eval_metrics['eval_loss']
    
    # Perform hyperparameter tuning using Optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(_objective, n_trials=n_trials)
    
    best_trial = study.best_trial
    best_params = best_trial.params
    
    return best_params


def main(args):
    # Load the model and tokenizer
    model_name = "microsoft/phi-1_5"
    model, tokenizer = load_pretrained_model(model_name)
    
    # Load and split fine-tuning dataset
    dataset = load_dataset("vicgalle/alpaca-gpt4")
    tokenized_datasets = dataset.map(
        partial(tokenize_line, tokenizer=tokenizer), batched=True
    )
    train_test_split = tokenized_datasets["train"].train_test_split(
        test_size=0.1
    )
    train_tokenized = train_test_split["train"]
    test_tokenized = train_test_split["test"]

    # Tune hyperparameters
    if args.tune_hyperparameters:
        best_params = tune_hyperparameters(
            model_name, args.lora_implementation, train_tokenized, 
            test_tokenized, n_trials=args.n_trials
        )
        lr, rank, alpha = \
            best_params["learning_rate"], best_params["r"], best_params["alpha"]
    else:
        lr, rank, alpha = args.learning_rate, args.rank, args.alpha
    
    # Fine-tune the model using full dataset
    full_tokenized = tokenized_datasets["train"]
    if args.lora_implementation == "peft":
        model = load_peft_lora(model, rank, alpha)
    elif args.lora_implementation == "custom":
        model = load_custom_lora(model, rank, alpha)
    else:
        raise ValueError(
            "Invalid LoRA implementation. Choose 'peft' or 'custom'."
        )
    trainer = train_model(model, lr, tokenizer, full_tokenized, full_tokenized,
                          output_dir="final_model")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning")
    parser.add_argument("--lora_implementation", type=str, default="peft", 
                        choices=["peft", "custom"])
    
    # Hyperparameter tuning
    parser.add_argument("--tune_hyperparameters", action="store_true")
    parser.add_argument("--n_trials", type=int, default=10)
    
    # Hyperparameter default values
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=16)
    
    args = parser.parse_args()
    
    main(args)
import argparse
import json
import logging
import os
import sys
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb
from jiwer import wer, cer
import wandb
import boto3
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Initialize S3 client
s3_client = boto3.client("s3")

class PhonemeDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        entry = self.data[idx]
        phonemes = ' '.join(entry['phonemes'])  # e.g., "AE Z Y UW K L AY M TH R UW"
        text = entry['text'].strip().lower()  # e.g., "as you climb through 3,000 feet"
        return {
            "phonemes": phonemes,
            "text": text,
            "input_text": f"[PHONEME] {phonemes} [/PHONEME] {text}"
        }

def collate_fn(batch):
    return {
        "phonemes": [item["phonemes"] for item in batch],
        "text": [item["text"] for item in batch],
        "input_text": [item["input_text"] for item in batch]
    }

def setup_wandb(args):
    logger.info("Logging in to W&B")
    secret_name = "wandb"
    region_name = "us-east-1"
    session = boto3.session.Session()
    secretsmanager = session.client(service_name='secretsmanager', region_name=region_name)
    try:
        get_secret_value_response = secretsmanager.get_secret_value(SecretId=secret_name)
        secret = get_secret_value_response['SecretString']
        api_key = json.loads(secret)["API_KEY"]
        wandb.login(key=api_key)
        wandb.init(project=args.project_name or "vallr-phoneme-llama")
    except Exception as e:
        logger.error(f"Failed to setup W&B: {str(e)}")

def validate_model(model, tokenizer, data_loader, device, max_length=50):
    model.eval()
    cer_scores = []
    wer_scores = []
    with torch.no_grad():
        for batch in data_loader:
            phonemes = batch["phonemes"]
            expected_texts = batch["text"]
            for phoneme, expected in zip(phonemes, expected_texts):
                inputs = tokenizer(f"[PHONEME] {phoneme} [/PHONEME]", return_tensors="pt").to(device)
                outputs = model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    num_beams=5,
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )
                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                if expected and generated:
                    cer_scores.append(cer(expected, generated))
                    wer_scores.append(wer(expected, generated))
    avg_cer = sum(cer_scores) / len(cer_scores) if cer_scores else 1.0
    avg_wer = sum(wer_scores) / len(wer_scores) if wer_scores else 1.0
    return avg_cer, avg_wer

def train(args, device):
    is_distributed = len(args.hosts) > 1 and args.backend is not None
    logger.debug(f"Distributed training - {is_distributed}")
    use_cuda = args.num_gpus > 0
    logger.debug(f"Number of GPUs available - {args.num_gpus}")

    if is_distributed:
        world_size = len(args.hosts)
        os.environ["WORLD_SIZE"] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)
        logger.info(f"Initialized distributed environment: '{args.backend}' backend, rank {host_rank}")

    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    # Load model and tokenizer
    model_name = "meta-llama/Llama-3.2-3B-Instruct"  # Or "bbunzeck/phoneme-llama"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ.get("HF_TOKEN"))
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        device_map="auto",
        token=os.environ.get("HF_TOKEN")
    )

    # Apply LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if is_distributed and use_cuda:
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        model = torch.nn.DataParallel(model)

    # Load dataset
    dataset = PhonemeDataset(os.path.join(args.train_data_dir, "annotations_subset.json"))
    hf_dataset = HFDataset.from_list(dataset.data)
    tokenized_dataset = hf_dataset.map(
        lambda x: tokenizer(x["input_text"], padding="max_length", truncation=True, max_length=128),
        batched=True,
        remove_columns=["phonemes", "text", "input_text"]
    )
    train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    # Validation DataLoader
    val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=200,
        logging_steps=50,
        learning_rate=args.lr,
        fp16=True if use_cuda else False,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        report_to="wandb",
        ddp_find_unused_parameters=False if is_distributed else None
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )

    # Train
    trainer.train()

    # Validate
    val_cer, val_wer = validate_model(model, tokenizer, val_loader, device)
    wandb.log({"val_cer": val_cer, "val_wer": val_wer})
    logger.info(f"Validation CER: {val_cer:.4f}, WER: {val_wer:.4f}")

    # Test sample
    test_phonemes = "AE Z Y UW K L AY M TH R UW"
    input_text = f"[PHONEME] {test_phonemes} [/PHONEME]"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs["input_ids"],
        max_length=50,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Test input: {test_phonemes}")
    logger.info(f"Generated text: {generated_text}")

    # Save model
    save_model(model, tokenizer, args.model_dir)

def save_model(model, tokenizer, model_dir):
    logger.info("Saving the model to S3")
    path = os.path.join(model_dir, "finetuned_model")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    wandb.save(path)
    logger.info("Model saved to W&B")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--backend", type=str, default=None, help="Distributed backend")
    parser.add_argument("--project-name", type=str, default="vallr-phoneme-llama", help="W&B project name")
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train-data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    args = parser.parse_args()

    setup_wandb(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(args, device)
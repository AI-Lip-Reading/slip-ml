import argparse
import json
import logging
import os
import sys
import torch
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
import re

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Initialize S3 client
s3_client = boto3.client("s3")

# Phoneme vocabulary
PHONEME_VOCAB = ['<blank>', 'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 
                 'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 
                 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 
                 'Y', 'Z', 'ZH']
IDX_TO_PHONEME = {i: p for i, p in enumerate(PHONEME_VOCAB)}

class PhonemeDataset(Dataset):
    def __init__(self, json_dir, output_file="phoneme_sentence_pairs.json"):
        self.data = []
        json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
        skipped = 0
        phoneme_lengths = []
        text_lengths = []

        for json_file in json_files:
            with open(os.path.join(json_dir, json_file), "r") as f:
                try:
                    label_data = json.load(f)
                    phoneme_indices = label_data.get("phoneme_indices", [])
                    text = label_data.get("text", "").strip().lower()
                    if not phoneme_indices or not text:
                        skipped += 1
                        continue
                    phonemes = [IDX_TO_PHONEME.get(idx, '<unknown>') for idx in phoneme_indices]
                    if '<unknown>' in phonemes:
                        skipped += 1
                        continue
                    phoneme_str = ' '.join(phonemes)
                    self.data.append({
                        "phonemes": phoneme_str,
                        "text": text,
                        "input_text": f"{phoneme_str} -> {text}"
                    })
                    phoneme_lengths.append(len(phonemes))
                    text_lengths.append(len(text.split()))
                except Exception as e:
                    logger.warning(f"Error processing {json_file}: {str(e)}")
                    skipped += 1

        with open(output_file, "w") as f:
            json.dump(self.data, f)
        logger.info(f"Processed {len(self.data)} samples, skipped {skipped}")
        logger.info(f"Avg phoneme length: {np.mean(phoneme_lengths):.2f}, Avg text length: {np.mean(text_lengths):.2f}")

        s3_key = os.path.join("data", output_file)
        s3_client.upload_file(output_file, args.bucket, s3_key)
        logger.info(f"Uploaded dataset to s3://{args.bucket}/{s3_key}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

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

def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return ' '.join(text.split())

def check_model_devices(model):
    devices = set()
    # Check parameters
    for param in model.parameters():
        devices.add(param.device)
    # Check buffers
    for buffer in model.buffers():
        devices.add(buffer.device)
    # Check bitsandbytes quantized parameters
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and hasattr(module.weight, 'device'):
            devices.add(module.weight.device)
    if len(devices) > 1:
        logger.error(f"Model components on multiple devices: {devices}")
        raise RuntimeError(f"Model components must be on a single device, found {devices}")
    logger.info(f"All model components on device: {devices.pop()}")

def validate_model(model, tokenizer, data_loader, max_length=256):
    model.eval()
    cer_scores = []
    wer_scores = []
    batch_size = data_loader.batch_size

    with torch.no_grad():
        for batch in data_loader:
            phonemes = batch["phonemes"]
            expected_texts = batch["text"]
            inputs = tokenizer(
                [f"{phoneme} ->" for phoneme in phonemes],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(model.device)
            logger.debug(f"Validation inputs device: {inputs['input_ids'].device}")
            outputs = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                num_beams=5,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
            generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for expected, generated in zip(expected_texts, generated_texts):
                expected = normalize_text(expected)
                generated = normalize_text(generated.split('->', 1)[-1].strip() if '->' in generated else generated)
                if expected and generated:
                    cer_scores.append(cer(expected, generated))
                    wer_scores.append(wer(expected, generated))

    avg_cer = sum(cer_scores) / len(cer_scores) if cer_scores else 1.0
    avg_wer = sum(wer_scores) / len(wer_scores) if wer_scores else 1.0
    return avg_cer, avg_wer

def train(args, device):
    use_cuda = torch.cuda.is_available()
    num_gpus = 1  # Force single-GPU training
    logger.info(f"Using single-GPU training on device {device}")
    logger.info(f"SM_HOSTS: {args.hosts}, SM_NUM_GPUS: {args.num_gpus}, SM_CURRENT_HOST: {args.current_host}")

    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    # Load model and tokenizer
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ.get("HF_TOKEN"))
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        token=os.environ.get("HF_TOKEN")
    ).to(device)
    logger.info(f"Model device after loading: {next(model.parameters()).device}")

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
    model.to(device)
    model.print_trainable_parameters()
    check_model_devices(model)

    # Load dataset
    dataset = PhonemeDataset(args.train_data_dir, output_file="phoneme_sentence_pairs.json")
    if len(dataset) == 0:
        raise ValueError("No valid data found in dataset")

    hf_dataset = HFDataset.from_list(dataset.data)
    logger.info(f"Initial dataset columns: {hf_dataset.column_names}")

    def tokenize_and_add_labels(examples):
        tokenized = tokenizer(
            examples["input_text"],
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="np"
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = hf_dataset.map(
        tokenize_and_add_labels,
        batched=True,
        remove_columns=["phonemes", "text", "input_text"]
    )
    logger.info(f"Tokenized dataset columns: {tokenized_dataset.column_names}")

    train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]
    logger.info(f"Train dataset columns: {train_dataset.column_names}, Eval dataset columns: {eval_dataset.column_names}")

    val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_steps=25,
        learning_rate=args.lr,
        fp16=True if use_cuda else False,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        report_to="wandb",
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )

    trainer.train()

    val_cer, val_wer = validate_model(model, tokenizer, val_loader, max_length=256)
    wandb.log({"val_cer": val_cer, "val_wer": val_wer})
    logger.info(f"Validation CER: {val_cer:.4f}, WER: {val_wer:.4f}")

    test_phonemes = "AE Z Y UW K L AY M TH R UW"
    input_text = f"{test_phonemes} ->"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    logger.debug(f"Test input device: {inputs['input_ids'].device}")
    outputs = model.generate(
        inputs["input_ids"],
        max_length=256,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text = generated_text.split('->', 1)[-1].strip() if '->' in generated_text else generated_text
    logger.info(f"Test input: {test_phonemes}")
    logger.info(f"Generated text: {generated_text}")

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
    parser.add_argument("--epochs", type=int, default=7, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--project-name", type=str, default="vallr-phoneme-llama", help="W&B project name")
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ.get("SM_HOSTS", "[]")))
    parser.add_argument("--current-host", type=str, default=os.environ.get("SM_CURRENT_HOST", "localhost"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train-data-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
    parser.add_argument("--num-gpus", type=int, default=1, help="Force single-GPU training")
    parser.add_argument("--bucket", type=str, default=os.environ.get("SM_DEFAULT_BUCKET", "slip-ml"), help="S3 bucket")
    args = parser.parse_args()

    setup_wandb(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train(args, device)
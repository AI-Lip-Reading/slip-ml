"""
LLaMA Fine-tuning with PyTorch FSDP (Fully Sharded Data Parallel)

This script fine-tunes a LLaMA model for phoneme-to-text conversion using PyTorch's FSDP.
FSDP is an advanced distributed training technique that shards model parameters, gradients,
and optimizer states across GPUs, enabling efficient training of large models.

Key FSDP features implemented:
- Mixed precision training (BFloat16/FP16)
- CPU offloading
- Activation checkpointing
- Parameter flattening
- Transformer auto wrapping
- Forward/backward prefetching
- Configurable sharding strategy (Zero2/Zero3)
- Memory-efficient model saving

Command line arguments allow for fine-grained control of FSDP behavior.
"""

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
    AutoConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    get_scheduler,
    SchedulerType,
    LlamaConfig,
    TrainerCallback
)
import transformers
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb
from jiwer import wer, cer
import wandb
import boto3
import numpy as np
import re

# Import utility for clearing CUDA cache
try:
    from accelerate.utils.memory import clear_device_cache
except ImportError:
    # Fallback implementation if accelerate is not available
    def clear_device_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            logger.info("Cleared CUDA cache manually (accelerate not available)")

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    CPUOffload,
    StateDictType,
    FullStateDictConfig,
    ShardedStateDictConfig
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap
)
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, LocalOptimStateDictConfig
import functools
import math
import sys
import smdistributed.dataparallel.torch.torch_smddp



backend = "smddp"
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
dist.init_process_group(backend=backend)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Transformers version: {transformers.__version__}")

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

def validate_model(model, tokenizer, data_loader, max_length=256, device="cuda"):
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
            ).to(device)
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
    # Initialize distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    logger.info(f"Distributed training initialized. Rank: {rank}, World size: {world_size}, Backend: {backend}")

    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed + rank)
        torch.cuda.set_device(local_rank)

    # Load model and tokenizer
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    
    # Add version compatibility check
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Using backend: {backend}")
    
    # First load tokenizer (doesn't require GPU)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        token=os.environ.get("HF_TOKEN"),
        use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Load the Hugging Face model configuration
    config = AutoConfig.from_pretrained(model_name)
    config.use_cache = False  # Disable cache for training

    # Determine appropriate dtype for model
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    logger.info(f"Using {compute_dtype} as compute dtype")
    
    # Initialize the base model first
    logger.info(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=compute_dtype,
        token=os.environ.get("HF_TOKEN"),
        # Use device_map="auto" for single GPU, but not with FSDP
        device_map=None  # Let FSDP handle the device mapping
    )
    
    # Create and apply LoRA configuration with consistent dtype
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        # Explicitly set LoRA parameters to use same dtype as base model
        inference_mode=False,
    )

    # Apply LoRA to the model
    logger.info("Applying LoRA adapters to model")
    model = get_peft_model(model, lora_config)
    
    # Move to device BEFORE parameter synchronization
    device_id = local_rank
    model = model.to(f"cuda:{device_id}")
    logger.info(f"Moved model to device cuda:{device_id}")
    
    # Ensure all parameters have consistent dtype before FSDP wrapping
    for name, param in model.named_parameters():
        if param.dtype != compute_dtype:
            logger.warning(f"Converting parameter {name} from {param.dtype} to {compute_dtype}")
            param.data = param.data.to(compute_dtype)
    
    # For SMDDP, use process group broadcast for synchronization
    if dist.is_initialized():
        logger.info(f"Synchronizing model parameters across processes (rank {rank})")
        # Let SMDDP handle the synchronization
        for name, param in model.named_parameters():
            if param.requires_grad:
                dist.broadcast(param.data, src=0)
        
        # Wait for all processes
        dist.barrier()
        logger.info(f"Model parameters synchronized on rank {rank}")

    # Define precision policy based on available hardware
    bf16_ready = (
        torch.cuda.is_available() and
        torch.cuda.is_bf16_supported()
    )
    
    if bf16_ready:
        logger.info("Using BFloat16 mixed precision")
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16
        )
    else:
        logger.info("Using Float16 mixed precision")
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16
        )
    
    # CPU offload configuration (can help with large models)
    cpu_offload = CPUOffload(offload_params=args.cpu_offload)
    
    # # Transformer auto wrap policy for more efficient FSDP wrapping
    # if args.flatten_parameters:
    #     # Combined approach using both transformer policy and size-based policy
    #     auto_wrap_policy = functools.partial(
    #         transformer_auto_wrap_policy,
    #         transformer_layer_cls={LlamaDecoderLayer},
    #         # Additionally wrap based on parameter size
    #         min_num_params=args.min_params_to_wrap
    #     )
    # else:
        # Just use transformer-based wrapping
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer}
    )
    
    # Select sharding strategy based on arguments
    if args.sharding_strategy == "FULL_SHARD":
        sharding_strategy = ShardingStrategy.FULL_SHARD  # Zero3
    else:
        sharding_strategy = ShardingStrategy.SHARD_GRAD_OP  # Zero2
    
    # Select backward prefetch strategy
    if args.backward_prefetch == "BACKWARD_PRE":
        backward_prefetch = BackwardPrefetch.BACKWARD_PRE
    elif args.backward_prefetch == "BACKWARD_POST":
        backward_prefetch = BackwardPrefetch.BACKWARD_POST
    else:
        backward_prefetch = None  # No prefetching
    
    # FSDP setup with advanced configurations
    fsdp_kwargs = {
        "auto_wrap_policy": auto_wrap_policy,
        "mixed_precision": mixed_precision_policy,
        "sharding_strategy": sharding_strategy,
        "limit_all_gathers": True,
        "cpu_offload": cpu_offload,
        "device_id": torch.cuda.current_device(),
        "use_orig_params": True,  # Better compatibility with HF models
    }
    
    # Add optional configurations
    if backward_prefetch is not None:
        fsdp_kwargs["backward_prefetch"] = backward_prefetch
    
    if args.forward_prefetch:
        fsdp_kwargs["forward_prefetch"] = True
    
    # Create FSDP model
    model = FSDP(model, **fsdp_kwargs)
    
    logger.info(f"Created FSDP model with configuration: {fsdp_kwargs}")

    # Apply activation checkpointing based on user settings
    if args.activation_checkpointing:
        # Activation checkpointing to save memory with different implementations
        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper,
            offload_to_cpu=True,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT
        )
        
        # Define which layers to apply checkpointing
        check_fn = lambda submodule: isinstance(submodule, LlamaDecoderLayer)
        
        # Apply activation checkpointing
        apply_activation_checkpointing(
            model, 
            checkpoint_wrapper_fn=non_reentrant_wrapper, 
            check_fn=check_fn
        )
        logger.info(f"Applied activation checkpointing to {model.__class__.__name__}")
    
    # Log memory usage stats
    if rank == 0:
        gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        logger.info(f"GPU memory allocated: {gpu_memory_allocated:.2f} GB")
        logger.info(f"GPU memory reserved: {gpu_memory_reserved:.2f} GB")
        logger.info(f"FSDP Strategy: {args.sharding_strategy}, CPU Offload: {args.cpu_offload}, Backward Prefetch: {args.backward_prefetch}")
        logger.info(f"Parameter Flattening: {args.flatten_parameters}, Activation Checkpointing: {args.activation_checkpointing}")
    
    logger.info(f"Model initialized with FSDP and activation checkpointing")
    
    # Optional: Print FSDP model structure for debugging
    if rank == 0:
        logger.debug(f"FSDP Model structure: {model}")

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
    
    # Ensure consistent sharding across processes for distributed training
    if dist.is_initialized() and dist.get_world_size() > 1:
        logger.info(f"Sharding datasets for distributed training (rank {dist.get_rank()} of {dist.get_world_size()})")
        # Make sure dataset is deterministically shuffled the same way on all ranks
        train_dataset = train_dataset.shuffle(seed=args.seed)
        eval_dataset = eval_dataset.shuffle(seed=args.seed)
        
        # Shard the datasets according to process rank and world size
        train_dataset = train_dataset.shard(
            num_shards=dist.get_world_size(),
            index=dist.get_rank()
        )
        eval_dataset = eval_dataset.shard(
            num_shards=dist.get_world_size(),
            index=dist.get_rank()
        )
        logger.info(f"Rank {dist.get_rank()} has {len(train_dataset)} training samples and {len(eval_dataset)} eval samples")
    
    logger.info(f"Train dataset columns: {train_dataset.column_names}, Eval dataset columns: {eval_dataset.column_names}")

    val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)

    # Scheduler
    num_update_steps_per_epoch = math.ceil(len(train_dataset) / (args.batch_size * world_size * args.gradient_accumulation_steps))
    max_train_steps = args.epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=max_train_steps * args.gradient_accumulation_steps,
    )

    # Training loop (using Trainer for compatibility with Hugging Face)
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        logging_steps=25,
        learning_rate=args.lr,
        fp16=True if not torch.cuda.is_bf16_supported() else False,
        bf16=torch.cuda.is_bf16_supported(),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        report_to="wandb",
        remove_unused_columns=False
    )

    # Add performance monitoring callback
    callbacks = [FSDPTrainingMonitorCallback(model, rank)]

        # Add these attributes to tell Trainer not to wrap the model again
    model.is_parallelizable = True
    model.model_parallel = True

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # Track initial memory usage
    log_fsdp_performance_metrics(model, rank, epoch=0, step=0)

    # Run training
    trainer.train()

    # Track final memory usage
    log_fsdp_performance_metrics(model, rank, epoch=args.epochs, step="final")

    if rank == 0:
        val_cer, val_wer = validate_model(model, tokenizer, val_loader, max_length=256, device=device)
        wandb.log({"val_cer": val_cer, "val_wer": val_wer})
        logger.info(f"Validation CER: {val_cer:.4f}, WER: {val_wer:.4f}")

        test_phonemes = "AE Z Y UW K L AY M TH R UW"
        input_text = f"{test_phonemes} ->"
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
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

    save_model(model, tokenizer, args.model_dir, rank)
    dist.barrier()

def save_model(model, tokenizer, model_dir, rank):
    logger.info("Saving the model with FSDP state dict")
    if rank == 0:
        path = os.path.join(model_dir, "finetuned_model")
        os.makedirs(path, exist_ok=True)
        
        try:
            # Configure FSDP state dict for memory-efficient saving
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            
            # Track memory before saving
            pre_save_mem = torch.cuda.memory_allocated() / (1024 ** 3)
            logger.info(f"Memory before model saving: {pre_save_mem:.2f} GB")
            
            # Clear CUDA cache before saving
            clear_device_cache()
            
            # Save using FSDP's state dict utility
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                logger.info("Getting full state dict")
                cpu_state = model.state_dict()
                logger.info(f"Saving model to {path}")
                
                # Save state dict (parameters)
                torch.save(cpu_state, os.path.join(path, "pytorch_model.bin"))
                
                # Save PEFT configuration for adapter loading
                if hasattr(model, "peft_config"):
                    logger.info("Saving PEFT adapter configuration")
                    model.peft_config.save_pretrained(path)
                    
                    # For LoRA models, try to save in the PEFT format as well
                    try:
                        logger.info("Attempting to save with PEFT utilities")
                        unwrapped_model = model
                        while hasattr(unwrapped_model, "module"):
                            unwrapped_model = unwrapped_model.module
                            
                        if hasattr(unwrapped_model, "save_pretrained"):
                            # Use PEFT's native save method if available
                            unwrapped_model.save_pretrained(path)
                            logger.info("Model saved with PEFT's save_pretrained")
                    except Exception as peft_save_error:
                        logger.warning(f"Could not save with PEFT utilities: {str(peft_save_error)}")
                
                # Save tokenizer
                tokenizer.save_pretrained(path)
                
                # Log the save to W&B
                wandb.save(path)
                logger.info("Model saved to W&B")
            
            # Track memory after saving
            post_save_mem = torch.cuda.memory_allocated() / (1024 ** 3)
            logger.info(f"Memory after model saving: {post_save_mem:.2f} GB")
                
            # Upload to S3
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                if os.path.isfile(file_path):
                    s3_key = os.path.join("models", "finetuned_model", filename)
                    s3_client.upload_file(file_path, args.bucket, s3_key)
            logger.info(f"Model uploaded to S3: s3://{args.bucket}/models/finetuned_model/")
        
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            
            # Fallback method if the above fails
            logger.info("Attempting to save with sharded state dict as fallback")
            try:
                sharded_save_policy = ShardedStateDictConfig(offload_to_cpu=True)
                with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, sharded_save_policy):
                    state_dict = model.state_dict()
                    torch.save(state_dict, os.path.join(path, "sharded_model.pt"))
                    tokenizer.save_pretrained(path)
                    logger.info("Successfully saved with sharded state dict")
            except Exception as inner_e:
                logger.error(f"Failed to save with sharded state dict: {str(inner_e)}")
    
    # Make sure all ranks wait for rank 0 to finish saving
    dist.barrier()

def log_fsdp_performance_metrics(model, rank, epoch=None, step=None):
    """
    Log performance metrics for FSDP model
    """
    if rank != 0:
        return  # Only log on rank 0
        
    prefix = f"Epoch {epoch}, Step {step}: " if epoch is not None and step is not None else ""
    
    # Memory metrics
    gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    gpu_max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
    
    # Log to logger
    logger.info(f"{prefix}GPU memory allocated: {gpu_memory_allocated:.2f} GB")
    logger.info(f"{prefix}GPU memory reserved: {gpu_memory_reserved:.2f} GB")
    logger.info(f"{prefix}GPU max memory: {gpu_max_memory:.2f} GB")
    
    # Log to wandb if it's available
    try:
        wandb.log({
            f"gpu/memory_allocated_gb": gpu_memory_allocated,
            f"gpu/memory_reserved_gb": gpu_memory_reserved,
            f"gpu/max_memory_gb": gpu_max_memory,
        })
    except Exception as e:
        logger.warning(f"Failed to log to wandb: {str(e)}")
    
    # Reset peak stats for next measurement
    torch.cuda.reset_peak_memory_stats()

class FSDPTrainingMonitorCallback(TrainerCallback):
    """Custom callback to monitor FSDP training performance"""
    
    def __init__(self, model, rank):
        self.model = model
        self.rank = rank
        self.step = 0
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each step"""
        self.step += 1
        # Log every 100 steps to avoid too much output
        if self.step % 100 == 0:
            log_fsdp_performance_metrics(self.model, self.rank, epoch=state.epoch, step=self.step)
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch"""
        log_fsdp_performance_metrics(self.model, self.rank, epoch=state.epoch, step="epoch_end")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=25, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=7, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--project-name", type=str, default="vallr-phoneme-llama", help="W&B project name")
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ.get("SM_HOSTS", "[]")))
    parser.add_argument("--current-host", type=str, default=os.environ.get("SM_CURRENT_HOST", "localhost"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train-data-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs, overridden by distributed setup")
    parser.add_argument("--bucket", type=str, default=os.environ.get("SM_DEFAULT_BUCKET", "slip-ml"), help="S3 bucket")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of updates steps to accumulate")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear", help="The scheduler type to use")
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use")
    # FSDP specific arguments
    parser.add_argument("--sharding_strategy", type=str, default="FULL_SHARD", choices=["FULL_SHARD", "SHARD_GRAD_OP"], 
                        help="FSDP sharding strategy: FULL_SHARD (Zero3) or SHARD_GRAD_OP (Zero2)")
    parser.add_argument("--cpu_offload", action="store_true", help="Enable CPU offloading for FSDP")
    parser.add_argument("--forward_prefetch", action="store_true", default=True, help="Enable forward prefetching for FSDP")
    parser.add_argument("--backward_prefetch", type=str, default="BACKWARD_PRE", choices=["BACKWARD_PRE", "BACKWARD_POST", "NONE"],
                        help="Backward prefetch strategy for FSDP")
    parser.add_argument("--flatten_parameters", action="store_true", default=True, 
                        help="Enable parameter flattening for FSDP (improves performance)")
    parser.add_argument("--activation_checkpointing", action="store_true", default=True,
                        help="Enable activation checkpointing to save memory")
    parser.add_argument("--min_params_to_wrap", type=int, default=1e7, 
                        help="Minimum number of parameters for a layer to be wrapped with FSDP")
    args = parser.parse_args()

    
    setup_wandb(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(args, device)
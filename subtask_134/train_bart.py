import os
import argparse
import random
import json
import math
from tqdm import tqdm
import numpy as np
import torch
# import torch.nn as nn
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.dataloader import DataLoader

from transformers import BartTokenizer, BartForConditionalGeneration, AutoConfig
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from dataset_simmc import SimmcDataset, SimmcDataCollator

import logging
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def training_step(model, inputs, args, loss_fn):
        model.train()
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(args.device)

        # compute loss
        if "labels" in inputs:
            labels = inputs.pop("labels")
        
        # force training to ignore pad token
        output = model(**inputs, use_cache=False)
        logits = output.logits
        
        loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))

        
        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()

        return loss.detach()


def prediction_step(model, inputs, loss_fn, prediction_loss_only=True):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(args.device)

        if "labels" in inputs:
            labels = inputs.pop("labels")

        with torch.no_grad():
            # compute loss on predict data
            output = model(**inputs, use_cache=False)
            logits = output.logits
            
            loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
            
        return loss.mean().detach()


def train(model, args, train_dataset, eval_dataset, data_collator, tokenizer, config=None):
        # Create output directory if needed
        os.makedirs(args.output_dir, exist_ok=True)
        
        if config is None:
            config = model.config
        
        # vocab_size = config.vocab_size
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=config.pad_token_id)

        # Data loader and number of training steps
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset,
            batch_size=args.train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
        )

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=args.eval_batch_size,
            collate_fn=data_collator,
        )

        # Multi-gpu training (should be after apex fp16 initialization)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
            
        model = model.to(args.device)
        
        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
        num_train_epochs = math.ceil(args.num_train_epochs)

        # create optimizer and scheduler
        no_decay = ["bias", "LayerNorm.weight"]
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
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=max_steps)

        # Train
        total_train_batch_size =  args.train_batch_size * args.gradient_accumulation_steps

        num_examples = len(train_dataloader.dataset)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", num_examples)
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", args.train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", max_steps)

        train_loss = []
        best_eval_loss = float('inf')
        best_epoch = 0
        model.zero_grad()
        for epoch in range(num_train_epochs):
            steps_in_epoch = len(train_dataloader)
            
            for step, inputs in enumerate(tqdm(train_dataloader, desc="Training")):
                tr_loss = training_step(model, inputs, args, loss_fn)
                train_loss.append(tr_loss.item())

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    model.zero_grad()
                    
                    
            
            # evaluate
            eval_loss = []
            model.eval()
            for step, inputs in enumerate(tqdm(eval_dataloader, desc="Evaluation")):
                loss = prediction_step(model, inputs, loss_fn)
                eval_loss.append(loss.item())

            eval_loss = np.mean(eval_loss)
            print("Epoch:", epoch+1, "Train loss:", np.mean(train_loss), "Eval loss:", eval_loss)

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_epoch = epoch
                # model.save_pretrained(args.output_dir)
                # torch.save(model,args.output_dir+"/check_saving")
                torch.save(model,args.output_dir+"/pytorch_model.bin")
                tokenizer.save_pretrained(args.output_dir)
                print("Saving best model at epoch", epoch+1)

            if args.patience > 0 and (epoch - best_epoch) > args.patience:
                print("Early stopping")
                break
            

               
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data', help='Input data directory')
    parser.add_argument('--output_dir', type=str, default='ckpt', help='Output directory where the model predictions and checkpoints will be written')

    # Model arguments
    parser.add_argument('--config_name', type=str, default=None, help='Pretrained config name or path if not the same as model_name')
    parser.add_argument('--model_name_or_path', type=str, default='facebook/bart-large', help='Path to pretrained model or model identifier from huggingface.co/models')

    # Training arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed for initialization')
    parser.add_argument('--train_batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--eval_batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--num_train_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of updates steps to accumulate the gradients for')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum gradient norm (for gradient clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='The weight decay to apply (if not zero)')
    parser.add_argument('--patience', type=int, default=3, help='Number of epochs with no improvement for early stopping. Set to 0 to disable early stopping')
    args = parser.parse_args()

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    
    # set seed
    if args.seed is not None:
        set_seed(args.seed)

    # init model
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)

    # add special tokens
    special_tokens = None
    special_tokens_file = os.path.join(args.data_dir, "simmc2_special_tokens.json")
    if os.path.exists(special_tokens_file):
        with open(special_tokens_file, "r") as f:
            special_tokens = json.load(f)

        tokenizer.add_special_tokens(special_tokens)
    print("Vocab size:", len(tokenizer)) # 50281

    model = BartForConditionalGeneration.from_pretrained(args.model_name_or_path, config=config)
    if special_tokens is not None:
        model.resize_token_embeddings(len(tokenizer))
        model.vocab_size = len(tokenizer)

    # init dataset
    train_src_file = os.path.join(args.data_dir, "train_predict.txt")
    train_tgt_file = os.path.join(args.data_dir, "train_target.txt")
    eval_src_file = os.path.join(args.data_dir, "dev_predict.txt")
    eval_tgt_file = os.path.join(args.data_dir, "dev_target.txt")
    
    train_dataset = SimmcDataset(tokenizer, train_src_file, train_tgt_file, prefix=config.prefix or "", is_lm=False)
    eval_dataset = SimmcDataset(tokenizer, eval_src_file, eval_tgt_file, prefix=config.prefix or "", is_lm=False)
    data_collator = SimmcDataCollator(tokenizer)
    
    # run train
    train(model, args, train_dataset, eval_dataset, data_collator, tokenizer, config)
         
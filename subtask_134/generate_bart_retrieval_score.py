import argparse
import os
# import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import SequentialSampler
from torch.utils.data.dataloader import DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration, AutoConfig
from dataset_simmc import SimmcDataset, SimmcDataCollator

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_up_text(text, eos_token):
    text = text.replace("</s>", "").strip()
    if eos_token in text:
        text = text.split(eos_token, 1)[0].strip()
    return text


def prediction_step(model, inputs, loss_fn, loss_fn2, pad_tokens, prediction_loss_only=True):
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(args.device)

    if "labels" in inputs:
        labels = inputs.pop("labels")
    with torch.no_grad():
        output = model(**inputs, use_cache=False)
        logits = output.logits

        # loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
        loss = loss_fn(logits.transpose(1,2), labels)
        result=loss.sum(-1)/(labels!=pad_tokens).sum(-1)


    # return loss.mean().detach()
    return result.detach()


# def prediction_step(model, inputs, args):
# 	for k, v in inputs.items():
# 		if isinstance(v, torch.Tensor):
# 			inputs[k] = v.to(args.device)
#
# 	gen_kwargs = {
# 		"max_length": args.max_len,
# 		"num_beams": 4,
# 	}
# 	with torch.no_grad():
# 		generated_tokens = model.generate(
# 			inputs["input_ids"],
# 			attention_mask=inputs["attention_mask"],
# 			**gen_kwargs,
# 		)
#
# 	return generated_tokens


# def generate(model, args, test_dataset, data_collator, tokenizer):
# 	os.makedirs(args.output_dir, exist_ok=True)
#
# 	test_sampler = SequentialSampler(test_dataset)
# 	test_dataloader = DataLoader(
# 		test_dataset,
# 		sampler=test_sampler,
# 		batch_size=args.eval_batch_size,
# 		collate_fn=data_collator,
# 	)
#
# 	# multi-gpu eval
# 	if args.n_gpu > 1:
# 		model = torch.nn.DataParallel(model)
#
# 	model = model.to(args.device)
#
# 	batch_size = test_dataloader.batch_size
# 	num_examples = len(test_dataloader.dataset)
# 	logger.info("***** Running Generation *****")
# 	logger.info("  Num examples = %d", num_examples)
# 	logger.info("  Batch size = %d", batch_size)
#
# 	with open(os.path.join(args.output_dir, "predictions.txt"), "w") as f:
# 		model.eval()
# 		for step, inputs in enumerate(tqdm(test_dataloader, desc="Generation")):
#
# 			logits = prediction_step(model, inputs, args)
# 			# logits = (batch, seq len)
#
# 			decoded = tokenizer.batch_decode(logits, skip_special_tokens=False, clean_up_tokenization_spaces=True)
#
# 			for line in decoded:
# 				line = clean_up_text(line, tokenizer.eos_token)
# 				f.write(line + "\n")

def eval_score_retrieval(model, args, test_dataset, data_collator, tokenizer, config):
    os.makedirs(args.output_dir, exist_ok=True)

    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=args.eval_batch_size,
        collate_fn=data_collator,
    )

    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    model = model.to(args.device)

    batch_size = test_dataloader.batch_size
    num_examples = len(test_dataloader.dataset)
    logger.info("***** Running Reranking *****")
    logger.info("  Num examples = %d", num_examples)
    logger.info("  Batch size = %d", batch_size)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=config.pad_token_id, reduction='none')
    loss_fn2 = torch.nn.CrossEntropyLoss(ignore_index=config.pad_token_id)
    test_loss = []
    with open(os.path.join(args.output_dir, "retrieval_scores.txt"), "w") as f:
        model.eval()
        for step, inputs in enumerate(tqdm(test_dataloader, desc="Generation")):

            loss = prediction_step(model, inputs, loss_fn,loss_fn2, config.pad_token_id)
            test_loss.extend(loss.tolist())
            # test_loss.append(loss.item())
            # logits = (batch, seq len)

            # decoded = tokenizer.batch_decode(logits, skip_special_tokens=False, clean_up_tokenization_spaces=True)

        for loss in test_loss:
            f.write(str(loss) + "\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='preprocessed/fashion', help='Input data directory')
    # parser.add_argument('--domain', type=str, default='fashion', help='fashion or furniture domain')
    parser.add_argument('--output_dir', type=str, default='output/fashion',
                        help='Output directory where the model predictions and checkpoints will be written')
    parser.add_argument('--model_name_or_path', type=str, default='ckpt/fashion',
                        help='Path to pretrained model or model identifier from huggingface.co/models')
    parser.add_argument('--eval_batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--max_len', type=int, default=300, help='Maximum sequence length for generation')
    args = parser.parse_args()

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # args.n_gpu = torch.cuda.device_count()
    args.n_gpu = 1

    # init model
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)
    print("Vocab size:", len(tokenizer))

    # model = BartForConditionalGeneration.from_pretrained(args.model_name_or_path)
    model = torch.load(args.model_name_or_path + "pytorch_model.bin")
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    # model.load_state_dict(torch.load(args.model_name_or_path+"pytorch_model.bin"))
    # init dataset
    test_src_file = os.path.join(args.data_dir, "retrieval_devtest_predict.txt")
    test_tgt_file = os.path.join(args.data_dir, "retrieval_devtest_target.txt")
    test_dataset = SimmcDataset(tokenizer, test_src_file, test_tgt_file, prefix=model.config.prefix or "", is_lm=False)
    data_collator = SimmcDataCollator(tokenizer)

    # generate
    eval_score_retrieval(model, args, test_dataset, data_collator, tokenizer, config)

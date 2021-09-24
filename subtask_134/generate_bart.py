import argparse
import os
# import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import SequentialSampler
from torch.utils.data.dataloader import DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration
from dataset_simmc import SimmcDataset, SimmcDataCollator

import logging
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)


def clean_up_text(text, eos_token):
	text = text.replace("</s>", "").strip()
	if eos_token in text:
		text = text.split(eos_token, 1)[0].strip()
	return text

def prediction_step(model, inputs, args):
	for k, v in inputs.items():
		if isinstance(v, torch.Tensor):
			inputs[k] = v.to(args.device)

	gen_kwargs = {
		"max_length": args.max_len,
		"num_beams": 4,
	}
	with torch.no_grad():
		generated_tokens = model.generate(
				inputs["input_ids"],
				attention_mask=inputs["attention_mask"],
				**gen_kwargs,
			)

	return generated_tokens



def generate(model, args, test_dataset, data_collator, tokenizer):
	
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
	logger.info("***** Running Generation *****")
	logger.info("  Num examples = %d", num_examples)
	logger.info("  Batch size = %d", batch_size)

	with open(os.path.join(args.output_dir, "predictions.txt"), "w") as f:
		model.eval()
		for step, inputs in enumerate(tqdm(test_dataloader, desc="Generation")):
			
			logits = prediction_step(model, inputs, args)
			# logits = (batch, seq len)
			
			decoded = tokenizer.batch_decode(logits, skip_special_tokens=False, clean_up_tokenization_spaces=True)

			for line in decoded:
				line = clean_up_text(line, tokenizer.eos_token)
				f.write(line + "\n")
				


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='preprocessed/fashion', help='Input data directory')
	parser.add_argument('--domain', type=str, default='fashion', help='fashion or furniture domain')
	parser.add_argument('--output_dir', type=str, default='output/fashion', help='Output directory where the model predictions and checkpoints will be written')
	parser.add_argument('--model_name_or_path', type=str, default='ckpt/fashion', help='Path to pretrained model or model identifier from huggingface.co/models')
	parser.add_argument('--eval_batch_size', type=int, default=16, help='Batch size for evaluation')
	parser.add_argument('--max_len', type=int, default=300, help='Maximum sequence length for generation')
	args = parser.parse_args()

	args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	# args.n_gpu = torch.cuda.device_count()
	args.n_gpu = 1

	# init model
	tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)
	print("Vocab size:", len(tokenizer))

	# model = BartForConditionalGeneration.from_pretrained(args.model_name_or_path)
	model = torch.load(args.model_name_or_path+"pytorch_model.bin")
	if isinstance(model, torch.nn.DataParallel):
		model = model.module
	# model.load_state_dict(torch.load(args.model_name_or_path+"pytorch_model.bin"))
	# init dataset
	test_src_file = os.path.join(args.data_dir, "devtest_predict.txt")
	test_dataset = SimmcDataset(tokenizer, test_src_file, None, prefix=model.config.prefix or "", is_lm=False)
	data_collator = SimmcDataCollator(tokenizer)

	# generate
	generate(model, args, test_dataset, data_collator, tokenizer)

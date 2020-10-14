# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from datetime import datetime
import json
import logging
import os
import random
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    CamembertConfig,
    CamembertForMaskedLM,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vlm.data import CoLDataset, get_voken_feats
from vlm.param import process_args
from vlm.model import CoLBertConfig, CoLwithBert


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    "openai-gpt": (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "bert": (CoLBertConfig, CoLwithBert, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    "camembert": (CamembertConfig, CamembertForMaskedLM, CamembertTokenizer),
}


def load_and_cache_examples(args, tokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    return CoLDataset(file_path, args.tokenizer_name, tokenizer, args.block_size,
                      split_sent=args.split_sent, voken_dir=args.voken_dir,
                      suffix=args.voken_suffix,
                      verbose=(args.gpu == 0),
                      voken_ablation=args.voken_ablation)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def mask_tokens(tokens: torch.Tensor, vokens: torch.Tensor, tokenizer: PreTrainedTokenizer, args) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Notice that this function would have a side affect of manipulating the Tensor tokens.
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = tokens.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    if args.voken_labels == 'mask':
        vokens[~masked_indices] = -100
    elif args.voken_labels == 'nonmask':
        vokens[masked_indices] = -100
    elif args.voken_labels == 'all':
        pass
    else:
        assert "Do not support the voken loss of type %s" % args.voken_labels

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    tokens[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    tokens[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return tokens, labels, vokens


def train(args, train_dataset: CoLDataset, valid_dataset: CoLDataset,
          model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:
    set_seed(args)  # Added here for reproducibility

    """ Train the model """
    if args.gpu == 0:
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        tb_writer = SummaryWriter(args.output_dir + '/runs/' + current_time)

    args.train_batch_size = args.per_gpu_train_batch_size

    def col_collate(examples):
        tokens, vokens = zip(*examples)
        if tokenizer._pad_token is None:
            tokens = pad_sequence(tokens, batch_first=True)
        else:
            tokens = pad_sequence(tokens, batch_first=True, padding_value=tokenizer.pad_token_id)
        vokens = pad_sequence(vokens, batch_first=True, padding_value=-100)
        return tokens, vokens

    if args.shuffle:
        logger.info(f"Shuffle the dataset in training,"
                       f"GPU: {args.gpu},"
                       f"Rank: {args.rank},"
                       f"Total: {args.world_size}")
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=args.shuffle,
    )
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, shuffle=False, num_workers=0,
        batch_size=args.train_batch_size, collate_fn=col_collate, pin_memory=True
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        # args.num_train_epochs = 9595
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    if args.lamb:
        no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']
    else:
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
    if args.lamb:
        logger.info(f"Using LAMB Optimizer with max grad norm {args.max_grad_norm}")
        import apex
        optimizer = apex.optimizers.FusedLAMB(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=args.adam_epsilon,
            max_grad_norm=args.max_grad_norm
        )
    else:
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=args.learning_rate,
                          #betas=(0.9, 0.98),
                          eps=args.adam_epsilon)
    if args.gpu == 0:
        print(f"Optimized with lr: {optimizer.defaults['lr']}, total steps: {t_total},"
              f" warmup steps: {args.warmup_steps}, epsilon {optimizer.defaults['eps']},"
              f" beta: {optimizer.defaults['betas']}, weight decay {args.weight_decay}.")
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if (
        args.model_name_or_path
        and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
        and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        from apex.parallel import DistributedDataParallel as DDP
        model = DDP(model)
    else:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )

    # Allow not calculating the lm heads.
    if args.mlm_ratio == 0.:
        model.lm_head = None


    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * args.world_size
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    # Check if continuing training from a checkpoint
    # if args.model_name_or_path and os.path.exists(args.model_name_or_path):
    #     try:
    #         # set global_step to gobal_step of last saved checkpoint from model path
    #         checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
    #         epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
    #         steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
    #         logger.info("  Continuing training from checkpoint, will skip to saved global_step")
    #         logger.info("  Continuing training from epoch %d", epochs_trained)
    #     except ValueError:
    #         logger.info("  Do not load model from %s, restart training" % args.model_name_or_path)

    model_to_resize = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    assert model_to_resize.config.vocab_size == len(tokenizer)
    # model_to_resize.resize_token_embeddings(len(tokenizer))

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.gpu != 0
    )
    set_seed(args)  # Added here for reproducibility
    LOSS_NAMES = ['token_loss', 'voken_loss', 'total_loss']
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.gpu != 0)
        tr_loss, logging_loss = np.zeros(len(LOSS_NAMES)), 0.0
        model.zero_grad()
        for step, (tokens, vokens) in enumerate(epoch_iterator):
            token_inputs, token_labels, voken_labels = mask_tokens(tokens, vokens, tokenizer, args)
            token_inputs = token_inputs.to(args.device)
            token_labels = token_labels.to(args.device) if args.mlm_ratio != 0. else None
            voken_labels = voken_labels.to(args.device)
            # If some of the input is padded, then the attention mask is needed
            attention_mask = (token_inputs != tokenizer.pad_token_id)         # word_tokens --> 1, pad_token --> 0
            if attention_mask.all():
                attention_mask = None

            if epoch == 0 and step < 3 and args.gpu == 0:
                print()
                print("Token inputs:", token_inputs.shape, token_inputs[0])
                print("Token inputs (in str): ", tokenizer.convert_ids_to_tokens(token_inputs[0].cpu().numpy()))
                print("Attention Mask:", attention_mask)
                print("Token Labels: ", token_labels[0] if token_labels is not None else token_labels)
                print("Token Labels (in str): ", tokenizer.convert_ids_to_tokens(token_labels[0].cpu().numpy()) if token_labels is not None else token_labels)
                print("Voken Labels: ", voken_labels[0])
                print()

            model.train()
            outputs = model(token_inputs,
                            attention_mask=attention_mask,
                            masked_lm_labels=token_labels,
                            voken_labels=voken_labels)
            voken_loss = outputs[0]
            token_loss = outputs[1]

            if args.mlm_ratio == 0.:
                loss = voken_loss
            else:
                loss = voken_loss + args.mlm_ratio * token_loss

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # print(f"GPU: {args.gpu}, Global Step: {global_step + 1}, "
            #       f"Step: {step}, "
            #       f"Range: {train_dataset.get_item_info(step * args.world_size + args.gpu)}, "
            #       f"Loss: {loss.item()}, "
            #       f"Scaled Loss: {scaled_loss.item()}")

            tr_loss += np.array((token_loss.item() / args.gradient_accumulation_steps,
                                 voken_loss.item() / args.gradient_accumulation_steps,
                                 loss.item()))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0. and not args.lamb:
                    # Only clip the grad when it is valid and not using LAMB optimizer,
                    # because the LAMB optimizer already apply grad clipping
                    if args.fp16:
                        total_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                elif args.max_grad_norm <= 0. and step <= args.gradient_accumulation_steps:
                    logger.warning("Have not clipped the gradient because "
                                   "the max_grad_norm is set to %0.2f" % args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.gpu == 0 and args.logging_steps > 0 and (step + 1) % args.logging_steps == 0:
                    # Log metrics
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    if args.fp16:
                        try:
                            from apex.amp import _amp_state
                            tb_writer.add_scalar("loss_scale", _amp_state.loss_scalers[0]._loss_scale, global_step)
                            tb_writer.add_scalar("scaled_loss", scaled_loss.item(), global_step)
                        except ImportError:
                            logger.warning("Cannot import apex.amp._amp_state, "
                                           "would not state the loss_scale in the log")
                    if args.max_grad_norm > 0. and not args.lamb:  # Only clip the grad when it is valid
                        tb_writer.add_scalar("grad_norm", total_norm, global_step)
                    interval_loss = (tr_loss - logging_loss) / args.logging_steps
                    for loss_idx, loss_name in enumerate(LOSS_NAMES):
                        tb_writer.add_scalar(loss_name, interval_loss[loss_idx], global_step)
                    logging_loss = tr_loss.copy()

            if args.max_steps > 0 and global_step >= args.max_steps:
                break

            # if step == 200:
            #     break
            #
        # Save it each epoch
        if args.gpu == 0:
            # Save checkpoints
            checkpoint_name = "checkpoint-epoch%04d" % epoch
            save_model(args, checkpoint_name, model, tokenizer, optimizer, scheduler)

            # last_path = os.path.join(args.output_dir, 'checkpoint-last')
            # if os.path.exists(last_path):
            #     os.remove(last_path)
            # os.symlink(os.path.join(args.output_dir, checkpoint_name), last_path)

            # Evaluate the model
            for loss_idx, loss_name in enumerate(LOSS_NAMES):
                logger.info(" Training %s of Epoch %d: %0.4f" % (
                    loss_name, epoch, tr_loss[loss_idx] / len(train_dataloader)))

            if args.do_eval:
                logger.info(" Evaluation Results of Epoch %d: " % epoch)
                old_eval_batch_size = args.per_gpu_eval_batch_size
                while args.per_gpu_eval_batch_size > 0:
                    try:
                        results = evaluate(args, valid_dataset, model, tokenizer)
                        break
                    except RuntimeError as e:
                        args.per_gpu_eval_batch_size = int(args.per_gpu_eval_batch_size / 2)
                        print("HALVE THE BATCH SIZE in EVAL.")
                        if args.per_gpu_eval_batch_size == 0:
                            raise e
                        time.sleep(5)
                args.per_gpu_eval_batch_size = old_eval_batch_size

                for key, value in results.items():
                    tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    logger.info("\t %s: %0.4f" % (key, value))
                tb_writer.add_scalar("epoch", epoch, global_step)
                output_eval_file = os.path.join(args.output_dir, checkpoint_name, "eval_results.json")
                json.dump(results, open(output_eval_file, 'w'), sort_keys=True, indent=4)
            # Currently, only GPU 0 is responsible for the evaluation.
            # torch.cuda.empty_cache()
            # torch.distributed.barrier()
        else:
            pass
            # torch.cuda.empty_cache()
            # torch.distributed.barrier()

        if args.max_steps > 0 and global_step >= args.max_steps:
            epoch_iterator.close()
            train_iterator.close()
            break

    if args.gpu == 0:
        tb_writer.close()


def save_model(args, name, model, tokenizer, optimizer, scheduler):
    # Save model checkpoint
    output_dir = os.path.join(args.output_dir, name)
    os.makedirs(output_dir, exist_ok=True)
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    logger.info("Saving model checkpoint to %s", output_dir)

    # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    # logger.info("Saving optimizer and scheduler states to %s", output_dir)


def evaluate(args, eval_dataset: CoLDataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix="") -> Dict:
    torch.cuda.empty_cache() 
    # # Loop to handle MNLI double evaluation (matched, mis-matched)
    # eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size
    # Note that DistributedSampler samples randomly

    def col_collate(examples):
        tokens, vokens = zip(*examples)
        if tokenizer._pad_token is None:
            tokens = pad_sequence(tokens, batch_first=True)
        else:
            tokens = pad_sequence(tokens, batch_first=True, padding_value=tokenizer.pad_token_id)
        vokens = pad_sequence(vokens, batch_first=True, padding_value=-100)
        return tokens, vokens

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=col_collate
    )

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    total_token_loss = 0.0
    total_voken_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for tokens, vokens in tqdm(eval_dataloader, desc="Evaluating"):
        token_inputs, token_labels, voken_labels = mask_tokens(tokens, vokens, tokenizer, args)
        token_inputs = token_inputs.to(args.device)
        token_labels = token_labels.to(args.device) if args.mlm_ratio != 0 else None
        voken_labels = voken_labels.to(args.device)
        # If some of the input is padded, then the attention mask is needed
        attention_mask = (token_inputs != tokenizer.pad_token_id)  # word_tokens --> 1, pad_token --> 0
        if attention_mask.all():
            attention_mask = None

        with torch.no_grad():
            outputs = model(token_inputs,
                            attention_mask=attention_mask,
                            masked_lm_labels=token_labels,
                            voken_labels=voken_labels)
            voken_loss = outputs[0]
            token_loss = outputs[1]

            total_voken_loss += voken_loss.item()
            total_token_loss += token_loss.item()

        nb_eval_steps += 1

    total_token_loss = total_token_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(total_token_loss)).item()

    result = {"perplexity": perplexity,
              "voken_loss": total_voken_loss / nb_eval_steps}
    torch.cuda.empty_cache() 

    return result


def is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def main():
    args = process_args()
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    port = 9595
    while is_port_in_use(port):
        port += 1
    print("Use port", port)
    os.environ['MASTER_PORT'] = str(port)

    # Using all available gpus for multi-processing distributed
    args.gpus = torch.cuda.device_count()
    print("Use gpus ", list(range(args.gpus)))
    args.world_size = args.gpus * args.nodes
    mp.spawn(setup, nprocs=args.gpus, args=(args,))


def setup(gpu, args):
    if args.should_continue:
        args.model_name_or_path = 'checkpoint-last'

    # Setup CUDA, GPU & distributed training
    torch.cuda.set_device(gpu)
    device = torch.device("cuda", gpu)
    args.gpu = gpu                                  # Local device id.
    args.device = device                            # Local device object.
    args.rank = args.nr * args.gpus + gpu           # The gpu id in the world.
    torch.distributed.init_process_group(
        backend="nccl",
        init_method='env://',
        world_size=args.world_size,
        rank=args.rank
    )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.gpu == 0 else logging.WARN,
    )
    logger.warning(
        "Process GPU: %s, num_of_total_GPUs: %s, distributed training: True, 16-bits training: %s",
        args.gpu, args.gpus, args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and token
    # Barrier to make sure only the first process in distributed training
    # download model & vocabizer
    if gpu != 0:
        torch.distributed.barrier()

    # Use self-defined models, thus avoiding Auto***.
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    # Next, we will initialize the training process in the following order:
    #   1. tokenizer --> 2. dataset --> 3. config --> 4. model.
    # because A) dataset relies on the tokenizer.special_tokens.
    #         B) config relies on the dataset.voken_size.

    # Get Tokenizer
    if args.tokenizer_name:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new {} tokenizer. This is not supported, "
            "but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name".format(tokenizer_class.__name__)
        )

    assert args.block_size <= tokenizer.max_len

    # Barrier to make sure only the first process in distributed training process the dataset,
    # and the others will use the cache
    if gpu != 0:
        torch.distributed.barrier()
    train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
    valid_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)
    if gpu == 0:
        torch.distributed.barrier()

    # Assert the vokens are equal in valid and eval.
    valid_dataset.assert_equal_vokens(train_dataset)

    config_kwargs = {}
    if args.do_voken_reg or args.do_voken_ctr:
        assert args.voken_feat_dir is not None
        voken_feats = get_voken_feats(train_dataset, args.voken_feat_dir)
        config_kwargs['voken_dim'] = len(voken_feats[0])
        if gpu == 0:
            logger.info(f"Load voken feats from {args.voken_feat_dir}"
                        f"with {len(voken_feats)} features and dimension {len(voken_feats[0])}")

    # Get Config
    if args.config_name:
        config = config_class.from_pretrained(
            args.config_name,
            cache_dir=args.cache_dir,
            voken_size=train_dataset.voken_size,
            do_voken_cls=args.do_voken_cls,
            do_voken_reg=args.do_voken_reg,
            do_voken_ctr=args.do_voken_ctr,
            shared_head=args.shared_head,
            verbose=(args.gpu == 0),
            **config_kwargs
        )
    elif args.model_name_or_path:
        config = config_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "Why do you want the default config?? Please use --config_name or --model_name_or_path"
        )

    if args.model_name_or_path:
        logger.info(f"Training model from the weight {args.model_name_or_path}.")
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = model_class(config=config)

    if args.do_voken_reg or args.do_voken_ctr:
        voken_feats = torch.tensor(voken_feats)
        model.init_voken_feat_emb(voken_feats)

    model.to(args.device)

    # End of barrier to make sure only the first process waiting other processes
    if gpu == 0:
        torch.distributed.barrier()

    if args.model_name_or_path:
        if gpu == 0:
            logger.info("Evaluate the performance of the loaded model.")
            results = evaluate(args, valid_dataset, model, tokenizer)
            for key, value in results.items():
                logger.info("\t %s: %0.4f" % (key, value))
            torch.distributed.barrier()
        else:
            torch.distributed.barrier()

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train(args, train_dataset, valid_dataset, model, tokenizer)

    # Evaluation
    if args.do_eval and gpu == 0:
        results = evaluate(args, valid_dataset, model, tokenizer)
        for key, value in results.items():
            logger.info("\t %s: %0.4f" % (key, value))


if __name__ == "__main__":
    main()

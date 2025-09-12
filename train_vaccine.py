import argparse
import os

import numpy as np
import torch
import ujson as json
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from model import DocREModel
from utils import set_seed, collate_fn
from prepro import read_docred
from evaluation import to_official, official_evaluate
# import wandb
from tqdm import tqdm


def train(args, model, train_features, dev_features, test_features):
    # Initialize wandb (disabled)
    # wandb.init(project="atlop-vaccine", name="vaccine-pathogen-relation-extraction")
    
    def finetune(features, optimizer, scheduler, num_epoch, num_steps):
        best_score = -1
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        for epoch in range(int(num_epoch)):
            model.zero_grad()
            epoch_progress = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{int(num_epoch)}", leave=False)
            for step, batch in enumerate(epoch_progress):
                model.train()
                inputs = {'input_ids': batch[0].to(args.device),
                          'attention_mask': batch[1].to(args.device),
                          'labels': batch[2],
                          'entity_pos': batch[3],
                          'hts': batch[4],
                          }
                outputs = model(**inputs)
                loss = outputs[0] / args.gradient_accumulation_steps
                loss.backward()
                if step % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    num_steps += 1
                # wandb.log({"loss": loss.item()}, step=num_steps)
                epoch_progress.set_postfix({"loss": f"{loss.item():.4f}", "step": num_steps})
                if step == len(train_dataloader) - 1:
                    dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
                    print(dev_output)
                    if dev_score > best_score:
                        best_score = dev_score
                        torch.save(model.state_dict(), f'best_ign_f1.pt')
                    # wandb.log(dev_output, step=num_steps)

    new_layer = ["extractor", "bilinear"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": 1e-4},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    
    train_dataloader = DataLoader(train_features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    t_total = len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=t_total * args.warmup_ratio, num_training_steps=t_total)
    
    num_steps = 0
    set_seed(args)
    model.zero_grad()
    finetune(train_features, optimizer, scheduler, args.num_train_epochs, num_steps)


def evaluate(args, model, features, tag="dev"):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    eval_progress = tqdm(dataloader, desc=f"Evaluating {tag}", leave=False)
    for batch in eval_progress:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        with torch.no_grad():
            pred, *_ = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    ans = to_official(preds, features)
    if len(ans) > 0:
        best_f1, _, best_f1_ign, _ = official_evaluate(ans, args.data_dir)
    output = {
        tag + "_F1": best_f1 * 100,
        tag + "_F1_ign": best_f1_ign * 100,
    }
    return best_f1, output


def report(args, model, features):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        with torch.no_grad():
            pred, *_ = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    preds = to_official(preds, features)
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./dataset/vaccine_pathogen_docred", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", type=str)

    parser.add_argument("--train_file", default="train_annotated.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)

    parser.add_argument("--load_path", default="", type=str)

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=8, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_labels", default=2, type=int,
                        help="Max number of labels in prediction.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Proportion of training to perform linear learning rate warmup for")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=500, type=int,
                        help="")
    parser.add_argument("--save_steps", default=500, type=int,
                        help="")
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")
    parser.add_argument("--num_class", default=2, type=int,
                        help="Number of relation types in data.")
    parser.add_argument("--device", type=str, default="mps", help="Device to use (cpu, cuda, mps)")
    parser.add_argument("--n_gpu", type=int, default=1, help="Number of GPUs to use")
    
    args = parser.parse_args()

    device = torch.device(args.device)
    args.device = device

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    config.transformer_type = args.transformer_type

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )

    if args.transformer_type == "bert":
        config.cls_token_id = tokenizer.cls_token_id
        config.sep_token_id = tokenizer.sep_token_id
    elif args.transformer_type == "roberta":
        config.cls_token_id = tokenizer.cls_token_id
        config.sep_token_id = tokenizer.sep_token_id
        config.sep_token_2_id = tokenizer.convert_tokens_to_ids(["</s>"])[0]

    model_bert = AutoModel.from_pretrained(args.model_name_or_path, config=config)
    model = DocREModel(config, model_bert, num_labels=args.num_labels)

    train_file = os.path.join(args.data_dir, args.train_file)
    dev_file = os.path.join(args.data_dir, args.dev_file)
    test_file = os.path.join(args.data_dir, args.test_file)

    train_features = read_docred(train_file, tokenizer, max_seq_length=args.max_seq_length)
    dev_features = read_docred(dev_file, tokenizer, max_seq_length=args.max_seq_length)
    test_features = read_docred(test_file, tokenizer, max_seq_length=args.max_seq_length)

    model.to(device)

    if args.load_path == "":  # Training
        train(args, model, train_features, dev_features, test_features)
    else:  # Testing
        model.load_state_dict(torch.load(args.load_path))
        dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
        print(dev_output)
        pred = report(args, model, test_features)
        with open("result.json", "w") as fh:
            json.dump(pred, fh)


if __name__ == "__main__":
    main()

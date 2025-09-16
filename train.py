import argparse
import os

import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
import ujson as json
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from model import DocREModel
from utils import set_seed, collate_fn
from prepro import read_docred
from evaluation import to_official, official_evaluate
import wandb
from tqdm import tqdm
import pickle
import os


def train(args, model, train_features, dev_features, test_features):
    def finetune(features, optimizer, num_epoch, num_steps, scaler):
        best_score = -1
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        train_iterator = range(int(num_epoch))
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        print("Total steps: {}".format(total_steps))
        print("Warmup steps: {}".format(warmup_steps))
        
        # Create progress bar for epochs
        epoch_pbar = tqdm(train_iterator, desc="Training", unit="epoch")
        for epoch in epoch_pbar:
            model.zero_grad()
            epoch_pbar.set_description(f"Epoch {epoch+1}/{int(num_epoch)}")
            
            # Create progress bar for steps within each epoch
            step_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} Steps", leave=False)
            for step, batch in enumerate(step_pbar):
                model.train()
                inputs = {'input_ids': batch[0].to(args.device),
                          'attention_mask': batch[1].to(args.device),
                          'labels': batch[2],
                          'entity_pos': batch[3],
                          'hts': batch[4],
                          }
                
                with autocast():
                    outputs = model(**inputs)
                    loss = outputs[0] / args.gradient_accumulation_steps
                
                scaler.scale(loss).backward()
                
                if step % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    model.zero_grad()
                    num_steps += 1
                wandb.log({"loss": loss.item()}, step=num_steps)
                
                # Update step progress bar with loss
                step_pbar.set_postfix({"loss": f"{loss.item():.4f}", "step": num_steps})
                
                if (step + 1) == len(train_dataloader) - 1 or (args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
                    step_pbar.close()  # Close step progress bar before evaluation
                    dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
                    wandb.log(dev_output, step=num_steps)
                    print(dev_output)
                    if dev_score > best_score:
                        best_score = dev_score
                        pred = report(args, model, test_features)
                        with open("result.json", "w") as fh:
                            json.dump(pred, fh)
                        if args.save_path != "":
                            torch.save(model.state_dict(), args.save_path)
                    # Recreate step progress bar after evaluation
                    step_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} Steps", leave=False, initial=step+1)
            
            step_pbar.close()  # Close step progress bar at end of epoch
            epoch_pbar.set_postfix({"best_f1": f"{best_score:.4f}"})
        
        epoch_pbar.close()  # Close epoch progress bar
        return num_steps

    new_layer = ["extractor", "bilinear"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": 1e-4},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scaler = GradScaler()
    num_steps = 0
    set_seed(args)
    model.zero_grad()
    finetune(train_features, optimizer, args.num_train_epochs, num_steps, scaler)


def display_test_examples(args, model, test_features, num_examples=3):
    """Display a few test examples with their inputs and predicted logits"""
    print("\n" + "="*80)
    print("DISPLAYING TEST EXAMPLES WITH PREDICTED LOGITS")
    print("="*80)
    
    model.eval()
    dataloader = DataLoader(test_features[:num_examples], batch_size=1, shuffle=False, collate_fn=collate_fn, drop_last=False)
    
    for i, batch in enumerate(dataloader):
        print(f"\n--- EXAMPLE {i+1} ---")
        
        # Get the original test feature for this example
        test_feature = test_features[i]
        
        # Display input information
        print(f"Title: {test_feature.get('title', 'N/A')}")
        print(f"Number of entities: {len(test_feature['vertexSet'])}")
        print(f"Number of relations: {len(test_feature.get('labels', []))}")
        
        # Show entity information
        print("\nEntities:")
        for j, entity in enumerate(test_feature['vertexSet']):
            entity_text = " ".join([test_feature['sents'][pos[0]][pos[1]:pos[2]] for pos in entity])
            print(f"  Entity {j}: {entity_text}")
        
        # Show ground truth relations
        if 'labels' in test_feature:
            print("\nGround Truth Relations:")
            for label in test_feature['labels']:
                h_idx, t_idx, rel = label['h'], label['t'], label['r']
                h_entity = " ".join([test_feature['sents'][pos[0]][pos[1]:pos[2]] for pos in test_feature['vertexSet'][h_idx]])
                t_entity = " ".join([test_feature['sents'][pos[0]][pos[1]:pos[2]] for pos in test_feature['vertexSet'][t_idx]])
                print(f"  {h_entity} --[{rel}]--> {t_entity}")
        
        # Get model predictions
        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }
        
        with torch.no_grad():
            with autocast():
                pred, *_ = model(**inputs)
                pred = pred.cpu().numpy()
                pred[np.isnan(pred)] = 0
        
        # Display predicted logits
        print(f"\nPredicted Logits (shape: {pred.shape}):")
        print(f"  Logits: {pred[0]}")
        
        # Show predicted relations (assuming binary classification)
        if pred.shape[1] == 2:  # Binary classification
            probabilities = torch.softmax(torch.tensor(pred[0]), dim=0)
            print(f"  Probabilities: [No Relation: {probabilities[0]:.4f}, Relation: {probabilities[1]:.4f}]")
            predicted_class = "Relation" if probabilities[1] > 0.5 else "No Relation"
            print(f"  Predicted: {predicted_class}")
        
        print("-" * 60)


def evaluate(args, model, features, tag="dev"):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    for batch in tqdm(dataloader, desc=f"Evaluating {tag}", leave=False):
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
    for batch in tqdm(dataloader, desc="Generating predictions", leave=False):
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

    parser.add_argument("--data_dir", default="./dataset/docred", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)

    parser.add_argument("--train_file", default="train_annotated.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)
    parser.add_argument("--save_path", default="", type=str)
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
    parser.add_argument("--num_labels", default=4, type=int,
                        help="Max number of labels in prediction.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=97,
                        help="Number of relation types in dataset.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for training (cuda or cpu).")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="Directory to cache preprocessed datasets.")
    parser.add_argument("--use_cache", action="store_true",
                        help="Use cached preprocessed datasets if available.")
    parser.add_argument("--save_cache", action="store_true",
                        help="Save preprocessed datasets to cache.")

    args = parser.parse_args()
    wandb.init(project="DocRED")

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda:0")
        args.n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cpu")
        args.n_gpu = 0
    args.device = device

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )

    read = read_docred

    # Create cache directory if it doesn't exist
    if args.save_cache or args.use_cache:
        os.makedirs(args.cache_dir, exist_ok=True)

    # Generate cache filenames based on dataset and model parameters
    cache_suffix = f"_{args.transformer_type}_{args.max_seq_length}_{args.model_name_or_path.split('/')[-1]}"
    train_cache_file = os.path.join(args.cache_dir, f"train_features{cache_suffix}.pkl")
    dev_cache_file = os.path.join(args.cache_dir, f"dev_features{cache_suffix}.pkl")
    test_cache_file = os.path.join(args.cache_dir, f"test_features{cache_suffix}.pkl")

    # Load or process datasets
    if args.use_cache and os.path.exists(train_cache_file) and os.path.exists(dev_cache_file) and os.path.exists(test_cache_file):
        print("Loading cached preprocessed datasets...")
        with open(train_cache_file, 'rb') as f:
            train_features = pickle.load(f)
        with open(dev_cache_file, 'rb') as f:
            dev_features = pickle.load(f)
        with open(test_cache_file, 'rb') as f:
            test_features = pickle.load(f)
        print("Cached datasets loaded successfully!")
    else:
        print("Processing datasets (this may take a while)...")
        train_file = os.path.join(args.data_dir, args.train_file)
        dev_file = os.path.join(args.data_dir, args.dev_file)
        test_file = os.path.join(args.data_dir, args.test_file)
        
        train_features = read(train_file, tokenizer, max_seq_length=args.max_seq_length)
        dev_features = read(dev_file, tokenizer, max_seq_length=args.max_seq_length)
        test_features = read(test_file, tokenizer, max_seq_length=args.max_seq_length)
        
        if args.save_cache:
            print("Saving preprocessed datasets to cache...")
            with open(train_cache_file, 'wb') as f:
                pickle.dump(train_features, f)
            with open(dev_cache_file, 'wb') as f:
                pickle.dump(dev_features, f)
            with open(test_cache_file, 'wb') as f:
                pickle.dump(test_features, f)
            print("Datasets cached successfully!")

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type

    set_seed(args)
    model = DocREModel(config, model, num_labels=args.num_labels)
    model.to(0)

    if args.load_path == "":  # Training
        train(args, model, train_features, dev_features, test_features)
        # Display test examples after training
        print("\n" + "="*80)
        print("TRAINING COMPLETED - DISPLAYING TEST EXAMPLES")
        print("="*80)
        display_test_examples(args, model, test_features, num_examples=3)
    else:  # Testing
        model.load_state_dict(torch.load(args.load_path))
        dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
        print(dev_output)
        pred = report(args, model, test_features)
        with open("result.json", "w") as fh:
            json.dump(pred, fh)
        # Display test examples after evaluation
        print("\n" + "="*80)
        print("EVALUATION COMPLETED - DISPLAYING TEST EXAMPLES")
        print("="*80)
        display_test_examples(args, model, test_features, num_examples=3)


if __name__ == "__main__":
    main()

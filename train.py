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


def train(args, model, train_features, dev_features, test_features, tokenizer):
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
                    loss = outputs['loss'] / args.gradient_accumulation_steps
                
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
                
                # Check if we should run evaluation
                is_last_step_of_epoch = (step + 1) == len(train_dataloader)
                is_evaluation_step = (args.evaluation_steps > 0 and 
                                    num_steps % args.evaluation_steps == 0 and 
                                    step % args.gradient_accumulation_steps == 0)
                
                if is_last_step_of_epoch or is_evaluation_step:
                    step_pbar.close()  # Close step progress bar before evaluation
                    dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
                    wandb.log(dev_output, step=num_steps)
                    print(dev_output)
                    
                    # Display test examples to monitor progress
                    print(f"\n{'='*60}")
                    print(f"TRAINING PROGRESS - TEST EXAMPLES (Step {num_steps})")
                    print(f"{'='*60}")
                    print(display_test_examples(args, model, test_features, tokenizer, num_examples=5))
                    
                    if dev_score > best_score:
                        best_score = dev_score
                        pred = report(args, model, test_features)
                        with open("result.json", "w") as fh:
                            json.dump(pred, fh)
                        if args.save_checkpoint != "":
                            torch.save(model.state_dict(), args.save_checkpoint)
                    # Recreate step progress bar after evaluation
                    step_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} Steps", leave=False, initial=step+1)
            
            step_pbar.close()  # Close step progress bar at end of epoch
            
            # Run evaluation at end of epoch to show all metrics
            print(f"\n{'='*80}")
            print(f"END OF EPOCH {epoch+1} - EVALUATION")
            print(f"{'='*80}")
            epoch_dev_score, epoch_dev_output = evaluate(args, model, dev_features, tag="dev")
            print(f"Epoch {epoch+1} Dev Metrics:")
            for key, value in epoch_dev_output.items():
                print(f"  {key}: {value:.4f}")
            print(f"  Best F1_ign so far: {best_score:.4f}")
            
            # Display test examples
            print(f"\n{'='*80}")
            print(f"END OF EPOCH {epoch+1} - DISPLAYING TEST EXAMPLES")
            print(f"{'='*80}")
            print(display_test_examples(args, model, test_features, tokenizer, num_examples=5))
            
            # Update progress bar with current epoch metrics
            epoch_pbar.set_postfix({
                "dev_f1": f"{epoch_dev_output['dev_F1']:.2f}",
                "dev_f1_ign": f"{epoch_dev_output['dev_F1_ign']:.2f}",
                "best_f1_ign": f"{best_score:.4f}"
            })
        
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


def display_test_examples(args, model, test_features, tokenizer, num_examples=1):
    """Display a few test examples with their inputs and predicted logits"""
    output_lines = []
    output_lines.append("\n" + "="*80)
    output_lines.append("DISPLAYING TEST EXAMPLES WITH PREDICTED LOGITS")
    output_lines.append("="*80)
    
    # Load relation mapping
    rel2id = json.load(open(args.data_dir + '/meta/rel2id.json', 'r'))
    id2rel = {v: k for k, v in rel2id.items()}
    
    model.eval()
    dataloader = DataLoader(test_features[:num_examples], batch_size=1, shuffle=False, collate_fn=collate_fn, drop_last=False)
    
    for i, batch in enumerate(dataloader):
        output_lines.append(f"\n--- EXAMPLE {i+1} ---")
        
        # Get the original test feature for this example
        test_feature = test_features[i]
        
        # Display input information
        output_lines.append(f"Title: {test_feature.get('title', 'N/A')}")
        output_lines.append(f"Number of entities: {len(test_feature['entity_pos'])}")
        output_lines.append(f"Number of entity pairs: {len(test_feature['hts'])}")
        
        # Show the original input text (without entity markers)
        input_tokens = tokenizer.convert_ids_to_tokens(test_feature['input_ids'])
        input_text = tokenizer.convert_tokens_to_string(input_tokens)
        
        # Remove entity markers for the initial display
        original_text = input_text.replace(' * ', ' ').replace('* ', '').replace(' *', '')
        output_lines.append(f"\nOriginal input text:")
        output_lines.append(f"'{original_text}'")
        output_lines.append(f"Input length: {len(test_feature['input_ids'])} tokens")
        
        # Load original test data to get correct entity names
        test_data_path = os.path.join(args.data_dir, "test.json")
        entity_names = []
        
        try:
            with open(test_data_path, 'r') as f:
                original_test_data = json.load(f)
            
            if i < len(original_test_data):
                original_doc = original_test_data[i]
                if 'vertexSet' in original_doc:
                    for entity_group in original_doc['vertexSet']:
                        if entity_group:  # Check if entity group is not empty
                            entity_names.append(entity_group[0]['name'])
        except:
            pass
        
        # Fallback to generic names if we can't load original data
        if not entity_names:
            entity_names = [f"Entity_{j}" for j in range(len(test_feature['entity_pos']))]
        
        output_lines.append(f"Entity names (from original data): {entity_names}")
        
        # Get model predictions
        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }
        
        with torch.no_grad():
            with autocast():
                outputs = model(**inputs)
                pred = outputs['processed_logits'].cpu().numpy()  # Processed predictions
                raw_pred = outputs['raw_logits'].cpu().numpy()  # Raw logits
                pred[np.isnan(pred)] = 0
                raw_pred[np.isnan(raw_pred)] = 0

        # Show the model-visible input (entities masked)
        masked_tokens = tokenizer.convert_ids_to_tokens(outputs['masked_input_ids'][0].cpu())
        masked_text = tokenizer.convert_tokens_to_string(masked_tokens)
        output_lines.append("Model-visible input (entities masked):")
        output_lines.append(f"'{masked_text}'")
        
        # Display predictions with ground truth - one pair at a time with context
        output_lines.append(f"\nEntity Pair Predictions:")
        output_lines.append("=" * 80)
        
        for pair_idx in range(len(pred)):
            h_idx, t_idx = test_feature['hts'][pair_idx]
            
            # Get ground truth label
            if 'labels' in test_feature and pair_idx < len(test_feature['labels']):
                gt_label = test_feature['labels'][pair_idx]
                if isinstance(gt_label, list) and len(gt_label) == 2:
                    # For binary classification, check which class is positive
                    gt_class = id2rel[1] if gt_label[1] > 0.5 else id2rel[0]
                else:
                    gt_class = "Unknown"
            else:
                gt_class = "Unknown"
            
            # Get prediction
            num_classes = raw_pred.shape[1]
            raw_logits = raw_pred[pair_idx]
            probabilities = torch.softmax(torch.tensor(raw_logits), dim=0)
            
            # Find predicted class (highest probability)
            predicted_class_id = torch.argmax(probabilities).item()
            predicted_class = id2rel.get(predicted_class_id, f"Class_{predicted_class_id}")
            confidence = probabilities[predicted_class_id].item()
            
            # Show entity names
            head_name = entity_names[h_idx] if h_idx < len(entity_names) else f"Entity_{h_idx}"
            tail_name = entity_names[t_idx] if t_idx < len(entity_names) else f"Entity_{t_idx}"
            
            output_lines.append(f"\n--- PAIR {pair_idx+1} ---")
            output_lines.append(f"Head Entity: {head_name}")
            output_lines.append(f"Tail Entity: {tail_name}")
            output_lines.append(f"Ground Truth: {gt_class}")
            output_lines.append(f"Predicted: {predicted_class}")
            output_lines.append(f"Confidence: {confidence:.3f}")
            
            # Show raw logits for all classes
            logits_str = "[" + ", ".join([f"{logit:.3f}" for logit in raw_logits]) + "]"
            output_lines.append(f"Raw Logits: {logits_str}")
            
            output_lines.append("-" * 60)
    
    return "\n".join(output_lines)


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
            outputs = model(**inputs)
            pred = outputs['processed_logits'].cpu().numpy()  # Use processed predictions for evaluation
            pred[np.isnan(pred)] = 0
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    ans = to_official(preds, features)
    if len(ans) > 0:
        best_f1, _, best_f1_ign, _ = official_evaluate(ans, args.data_dir, split=tag)
    output = {
        tag + "_F1": best_f1 * 100,
        tag + "_F1_ign": best_f1_ign * 100,
    }
    return best_f1_ign, output  # Use F1_ign for checkpointing


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
            outputs = model(**inputs)
            pred = outputs['processed_logits'].cpu().numpy()  # Use processed predictions for report
            pred[np.isnan(pred)] = 0
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    preds = to_official(preds, features)
    return preds


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./dataset/docred", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--base_model_name_or_path", default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", type=str,
                        help="Base model name or path (e.g., bert-base-cased, microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext)")

    parser.add_argument("--train_file", default="train_annotated.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)
    parser.add_argument("--save_checkpoint", default="", type=str,
                        help="Path to save the trained model checkpoint (e.g., best_model.pth)")
    parser.add_argument("--load_checkpoint", default="", type=str,
                        help="Path to load a pretrained model checkpoint (e.g., best_model.pth)")
    parser.add_argument("--fine_tune", action="store_true",
                        help="Fine-tune from a pretrained model (requires --load_checkpoint).")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=16, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_labels", default=97, type=int,
                        help="Max number of labels in prediction.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations. Set to -1 to disable evaluation during training.")
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
    parser.add_argument("--entity_masking", action="store_true",
                        help="Enable entity masking during training and inference.")
    parser.add_argument("--pretrain_on_docred", action="store_true",
                        help="Pretrain the model on DocRED.")
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
        args.config_name if args.config_name else args.base_model_name_or_path,
        num_labels=args.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.base_model_name_or_path,
    )
    
    read = read_docred

    # Create cache directory if it doesn't exist
    if args.save_cache or args.use_cache:
        os.makedirs(args.cache_dir, exist_ok=True)

    # Generate cache filenames based on dataset and model parameters
    dataset_name = os.path.basename(args.data_dir)  # e.g., "docred" or "vaccine_pathogen_docred"
    cache_suffix = f"_{dataset_name}_{args.transformer_type}_{args.max_seq_length}_{args.data_dir.split('/')[-1]}"
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
        args.base_model_name_or_path,
        from_tf=bool(".ckpt" in args.base_model_name_or_path),
        config=config,
    )

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type
    config.mask_token_id = tokenizer.mask_token_id
    config.entity_masking = args.entity_masking

    set_seed(args)
    model = DocREModel(config, model, num_labels=args.num_labels)
    model.to(0)

    if args.fine_tune:  # Fine-tuning from pretrained model
        if args.load_checkpoint == "":
            raise ValueError("--fine_tune requires --load_checkpoint to be specified")
        print(f"Loading pretrained model from {args.load_checkpoint}")
        model.load_state_dict(torch.load(args.load_checkpoint))
        print("Starting fine-tuning...")
        train(args, model, train_features, dev_features, test_features, tokenizer)
    elif args.load_checkpoint == "":  # Training from scratch
        train(args, model, train_features, dev_features, test_features, tokenizer)
    else:  # Testing/Evaluation only
        model.load_state_dict(torch.load(args.load_checkpoint))
        dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
        print("Dev evaluation results:")
        print(dev_output)
        
        # Test evaluation
        print("\n" + "="*80)
        print("TEST SET EVALUATION")
        print("="*80)
        test_score, test_output = evaluate(args, model, test_features, tag="test")
        print("Test evaluation results:")
        print(test_output)
        
        pred = report(args, model, test_features)
        # Write comprehensive test results
        test_results = {
            "dev_metrics": dev_output,
            "dev_score": dev_score,
            "test_metrics": test_output,
            "test_score": test_score,
            "test_predictions": pred,
            "display_results": {
                "num_test_examples": len(test_features),
                "num_predictions": len(pred),
                "model_info": {
                    "transformer_type": args.transformer_type,
                    "base_model_name": args.base_model_name_or_path,
                    "max_seq_length": args.max_seq_length,
                    "num_labels": args.num_labels,
                    "num_class": args.num_class,
                    "loaded_from": args.load_checkpoint
                }
            }
        }
        
        # Display test examples after evaluation
        print("\n" + "="*80)
        print("EVALUATION COMPLETED - DISPLAYING TEST EXAMPLES")
        print("="*80)
        test_examples_output = display_test_examples(args, model, test_features, tokenizer, num_examples=3)
        print(test_examples_output)
        
        # Save all results to a single text file
        with open("results_test_set.txt", "w") as fh:
            fh.write("TEST EVALUATION RESULTS\n")
            fh.write("="*80 + "\n")
            fh.write(f"Dev F1 Score: {dev_score:.4f}\n")
            fh.write(f"Test F1 Score: {test_score:.4f}\n")
            fh.write(f"Test Predictions: {len(pred)} relations predicted\n")
            fh.write(f"Number of test examples: {len(test_features)}\n")
            fh.write("\n")
            fh.write("TEST PREDICTIONS (JSON format):\n")
            fh.write("-" * 40 + "\n")
            fh.write(json.dumps(test_results, indent=2))
            fh.write("\n\n")
            fh.write("TEST EXAMPLES:\n")
            fh.write("-" * 40 + "\n")
            fh.write(test_examples_output)
        
        print(f"\nTest results written to results_test_set.txt")
        print(f"Dev F1 Score: {dev_score:.4f}")
        print(f"Test F1 Score: {test_score:.4f}")
        print(f"Test Predictions: {len(pred)} relations predicted")

if __name__ == "__main__":
    main()

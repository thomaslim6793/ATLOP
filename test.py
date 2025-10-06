import argparse
import os
import json
import pickle

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from tqdm import tqdm

from model import DocREModel
from utils import set_seed, collate_fn
from prepro import read_docred
from evaluation import to_official, official_evaluate


def evaluate(args, model, features, tag="dev"):
    # Load relation mapping
    rel2id = json.load(open(args.data_dir + '/meta/rel2id.json', 'r'))
    id2rel = {v: k for k, v in rel2id.items()}

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
    ans = to_official(preds, features, id2rel)
    if len(ans) > 0:
        best_f1, _, best_f1_ign, _ = official_evaluate(ans, args.data_dir, split=tag)
    output = {
        tag + "_F1": best_f1 * 100,
        tag + "_F1_ign": best_f1_ign * 100,
    }
    return best_f1_ign, output


def report(args, model, features):
    # Load relation mapping
    rel2id = json.load(open(args.data_dir + '/meta/rel2id.json', 'r'))
    id2rel = {v: k for k, v in rel2id.items()}

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
    preds = to_official(preds, features, id2rel)
    return preds


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
        
        # Show original input text
        input_tokens = tokenizer.convert_ids_to_tokens(batch[0][0].cpu())
        original_text = tokenizer.convert_tokens_to_string(input_tokens)
        output_lines.append(f"Original input: '{original_text}'")
        
        # Show entity information
        entity_positions = test_feature['entity_pos']
        output_lines.append(f"Number of entities: {len(entity_positions)}")
        
        # Show entity pair information
        hts = test_feature['hts']
        output_lines.append(f"Number of entity pairs: {len(hts)}")
        
        # Get entity names for display
        entity_names = []
        for entity_mentions in entity_positions:
            if entity_mentions:
                # Get the first mention of each entity
                start_pos = entity_mentions[0][0]
                end_pos = entity_mentions[0][1]
                if start_pos < len(input_tokens) and end_pos <= len(input_tokens):
                    entity_tokens = input_tokens[start_pos:end_pos]
                    # Remove asterisks from entity name display
                    entity_tokens_clean = [token for token in entity_tokens if token != "*"]
                    entity_name = tokenizer.convert_tokens_to_string(entity_tokens_clean)
                    entity_names.append(entity_name)
                else:
                    entity_names.append("UNKNOWN")
            else:
                entity_names.append("UNKNOWN")
        
        output_lines.append(f"Entity names: {entity_names}")
        
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
            if 'labels' in test_feature:
                ground_truth_labels = test_feature['labels'][pair_idx]
                # Find which relation is labeled as positive (1)
                positive_relations = [i for i, label in enumerate(ground_truth_labels) if label == 1]
                if positive_relations:
                    gt_relation = id2rel[positive_relations[0]]
                else:
                    gt_relation = "No relation"
            else:
                gt_relation = "Unknown"
            
            # Get predicted relation
            predicted_labels = pred[pair_idx]
            predicted_relation_idx = np.argmax(predicted_labels)
            predicted_relation = id2rel[predicted_relation_idx]
            confidence = predicted_labels[predicted_relation_idx]
            
            # Get raw logits for this prediction
            raw_logits = raw_pred[pair_idx]
            
            # Display prediction
            output_lines.append(f"\n--- PAIR {pair_idx + 1} ---")
            output_lines.append(f"Head Entity: {entity_names[h_idx] if h_idx < len(entity_names) else 'UNKNOWN'}")
            output_lines.append(f"Tail Entity: {entity_names[t_idx] if t_idx < len(entity_names) else 'UNKNOWN'}")
            output_lines.append(f"Ground Truth: {gt_relation}")
            output_lines.append(f"Predicted: {predicted_relation}")
            output_lines.append(f"Confidence: {confidence:.3f}")
            output_lines.append(f"Raw Logits: {raw_logits.tolist()}")
            output_lines.append("-" * 40)
    
    return "\n".join(output_lines)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./dataset/docred", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--base_model_name_or_path", default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", type=str,
                        help="Base model name or path (e.g., bert-base-cased, microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext)")

    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)
    parser.add_argument("--load_checkpoint", required=True, type=str,
                        help="Path to load a pretrained model checkpoint (e.g., best_model.pth)")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--test_batch_size", default=16, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--num_labels", default=None, type=int,
                        help="Max number of labels in prediction (optional, auto-inferred from rel2id.json if not specified).")
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=None,
                        help="Number of relation types in dataset (optional, auto-inferred from rel2id.json if not specified).")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for training (cuda or cpu).")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="Directory to cache preprocessed datasets.")
    parser.add_argument("--use_cache", action="store_true",
                        help="Use cached preprocessed datasets if available.")
    parser.add_argument("--entity_masking", action="store_true",
                        help="Enable entity masking during training and inference.")
    parser.add_argument("--display_test_examples", action="store_true",
                        help="Display test examples during evaluation.")
    
    args = parser.parse_args()

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda:0")
        args.n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cpu")
        args.n_gpu = 0
    args.device = device

    # Load relation mapping and auto-infer num_class BEFORE creating config
    rel2id_path = os.path.join(args.data_dir, 'meta', 'rel2id.json')
    rel2id = json.load(open(rel2id_path, 'r'))
    
    actual_num_class = len(rel2id)
    
    # Auto-infer num_class and num_labels if not specified
    if args.num_class is None:
        print(f"Auto-inferring num_class from rel2id.json: {actual_num_class}")
        args.num_class = actual_num_class
    elif args.num_class != actual_num_class:
        print(f"WARNING: --num_class={args.num_class} but rel2id.json has {actual_num_class} classes")
        print(f"Auto-adjusting num_class to {actual_num_class}")
        args.num_class = actual_num_class
    
    if args.num_labels is None:
        print(f"Auto-inferring num_labels from rel2id.json: {actual_num_class}")
        args.num_labels = actual_num_class
    elif args.num_labels != actual_num_class:
        print(f"WARNING: --num_labels={args.num_labels} but rel2id.json has {actual_num_class} classes")
        print(f"Auto-adjusting num_labels to {actual_num_class}")
        args.num_labels = actual_num_class

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.base_model_name_or_path,
        num_labels=args.num_class,
    )
    config.transformer_type = args.transformer_type

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.base_model_name_or_path,
    )

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type
    config.mask_token_id = tokenizer.mask_token_id

    # Load the base model
    model = AutoModel.from_pretrained(
        args.base_model_name_or_path,
        from_tf=bool(".ckpt" in args.base_model_name_or_path),
        config=config,
        cache_dir=args.cache_dir,
    )

    # Create cache directory if it doesn't exist
    if args.use_cache:
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
        print("Loading datasets from scratch...")
        train_features = read_docred(rel2id, os.path.join(args.data_dir, "train_annotated.json"), tokenizer, max_seq_length=args.max_seq_length)
        dev_features = read_docred(rel2id, os.path.join(args.data_dir, args.dev_file), tokenizer, max_seq_length=args.max_seq_length)
        test_features = read_docred(rel2id, os.path.join(args.data_dir, args.test_file), tokenizer, max_seq_length=args.max_seq_length)

    config.entity_masking = args.entity_masking

    set_seed(args)
    model = DocREModel(config, model, num_labels=args.num_labels)
    model.to(0)

    # Load the checkpoint
    print(f"Loading model from {args.load_checkpoint}")
    model.load_state_dict(torch.load(args.load_checkpoint))

    # Dev evaluation
    print("Running dev evaluation...")
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
    if args.display_test_examples:
        print("\n" + "="*80)
        print("EVALUATION COMPLETED - DISPLAYING TEST EXAMPLES")
        print("="*80)
        test_examples_output = display_test_examples(args, model, test_features, tokenizer, num_examples=1)
        print(test_examples_output)
    
    # Save all results to a single text file
    suffix = args.load_checkpoint.split("/")[-1].split(".")[0]
    file_name = f"results_test_set_{suffix}.txt"
    with open(file_name, "w") as fh:
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
        if args.display_test_examples:
            fh.write("\n\n")
            fh.write("TEST EXAMPLES:\n")
            fh.write("-" * 40 + "\n")
            fh.write(test_examples_output)
    
    print(f"\nTest results written to {file_name}")
    print(f"Dev F1 Score: {dev_score:.4f}")
    print(f"Test F1 Score: {test_score:.4f}")
    print(f"Test Predictions: {len(pred)} relations predicted")


if __name__ == "__main__":
    main()

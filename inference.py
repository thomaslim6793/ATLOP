"""
Vaccine-Pathogen Relation Extraction Inference Script
"""

import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
from model import DocREModel
from utils import collate_fn


def load_model(model_path="best_vaccine_model.pth"):
    """Load the trained vaccine-pathogen model"""
    print("Loading model...")
    
    # Load the base model and tokenizer
    model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name, num_labels=2)
    base_model = AutoModel.from_pretrained(model_name, config=config)
    
    # Set required config parameters
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = "bert"
    
    # Load custom model architecture
    model = DocREModel(config, base_model, num_labels=2)
    
    # Load trained weights
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    
    print("✅ Model loaded successfully!")
    return model, tokenizer


def preprocess_docred(json_input, tokenizer, rel2id, max_seq_length=1024):
    """Preprocess DocRED format input for inference"""
    features = []
    
    for sample in tqdm(json_input, desc="Preprocessing"):
        sents = []
        sent_map = []

        entities = sample['vertexSet']
        entity_start, entity_end = [], []
        for entity in entities:
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                entity_start.append((sent_id, pos[0]))
                entity_end.append((sent_id, pos[1] - 1))
        
        for i_s, sent in enumerate(sample['sents']):
            new_map = {}
            for i_t, token in enumerate(sent):
                tokens_wordpiece = tokenizer.tokenize(token)
                if (i_s, i_t) in entity_start:
                    tokens_wordpiece = ["*"] + tokens_wordpiece
                if (i_s, i_t) in entity_end:
                    tokens_wordpiece = tokens_wordpiece + ["*"]
                new_map[i_t] = len(sents)
                sents.extend(tokens_wordpiece)
            new_map[i_t + 1] = len(sents)
            sent_map.append(new_map)

        # Process entity positions
        entity_pos = []
        for e in entities:
            entity_pos.append([])
            for m in e:
                start = sent_map[m["sent_id"]][m["pos"][0]]
                end = sent_map[m["sent_id"]][m["pos"][1]]
                entity_pos[-1].append((start, end))

        # Create all possible entity pairs
        hts = []
        for h in range(len(entities)):
            for t in range(len(entities)):
                if h != t:
                    hts.append([h, t])

        # Tokenize
        sents = sents[:max_seq_length - 2]
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        feature = {
            'input_ids': input_ids,
            'entity_pos': entity_pos,
            'hts': hts,
            'title': sample['title'],
            'labels': None,
            'original_entities': sample['vertexSet']  # Add this line
        }
        features.append(feature)

    return features


def predict(model, preprocessed_inputs, device="cpu"):
    """Run inference on preprocessed inputs"""
    dataloader = DataLoader(preprocessed_inputs, batch_size=8, shuffle=False, collate_fn=collate_fn, drop_last=False)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    preds = []
    raw_logits = []
    
    for batch in tqdm(dataloader, desc="Predicting"):
        model.eval()
        
        inputs = {
            'input_ids': batch[0].to(device),
            'attention_mask': batch[1].to(device),
            'entity_pos': batch[3],
            'hts': batch[4],
        }

        with torch.no_grad():
            outputs = model(**inputs)
            pred = outputs['processed_logits'].cpu().numpy()
            raw = outputs['raw_logits'].cpu().numpy()
            pred[np.isnan(pred)] = 0
            raw[np.isnan(raw)] = 0
            preds.append(pred)
            raw_logits.append(raw)
    
    return np.concatenate(preds, axis=0), np.concatenate(raw_logits, axis=0)


def format_predictions(preds, raw_logits, features, rel2id, tokenizer):
    """Format predictions into user-friendly output"""
    id2rel = {v: k for k, v in rel2id.items()}
    results = []
    
    pred_idx = 0
    for feature in features:
        # Decode the input once per document (not per prediction)
        decoded_input = tokenizer.decode(feature['input_ids'], skip_special_tokens=False)
        
        doc_results = {
            'title': feature['title'],
            'input': decoded_input,  # Document-level input
            'predictions': []
        }
        
        # Get original entity names from vertexSet
        original_entities = feature['original_entities']
        entity_names = []
        for entity_group in original_entities:
            if entity_group:
                entity_names.append(entity_group[0]['name'])
            else:
                entity_names.append("Unknown_Entity")
        
        for h, t in feature['hts']:
            pred = preds[pred_idx]
            raw = raw_logits[pred_idx]
            
            # Get prediction
            predicted_relation = id2rel[pred.argmax()]
            
            # Calculate probabilities
            probabilities = torch.softmax(torch.tensor(raw), dim=0)
            
            result = {
                'head_entity': entity_names[h],
                'tail_entity': entity_names[t],
                'head_entity_type': original_entities[h][0].get('type', 'unknown') if original_entities[h] else 'unknown',
                'tail_entity_type': original_entities[t][0].get('type', 'unknown') if original_entities[t] else 'unknown',
                'predicted_relation': predicted_relation,
                'probabilities': {
                    id2rel[i]: float(probabilities[i]) for i in range(len(probabilities))
                },
                'raw_logits': raw.tolist()
                # Remove 'input' from here
            }
            doc_results['predictions'].append(result)
            pred_idx += 1
        
        results.append(doc_results)
    
    return results


def main():
    """Main inference function"""
    # Load model
    model, tokenizer = load_model("best_vaccine_model.pth")
    
    # Load relation mapping
    with open("meta/rel2id.json", "r") as f:
        rel2id = json.load(f)
    
    # Load input data
    with open("example_input.json", "r") as f:
        input_data = json.load(f)
    
    # Preprocess
    features = preprocess_docred(input_data, tokenizer, rel2id)
    
    # Predict
    preds, raw_logits = predict(model, features, device="cpu")
    
    # Format results
    results = format_predictions(preds, raw_logits, features, rel2id, tokenizer)
    
    # Display results
    for doc_result in results:
        print(f"\n📄 Document: {doc_result['title']}")
        print("="*60)
        print(f"🔤 Input: {doc_result['input']}")  # Show input once per document
        print("="*60)
        
        id2rel = {v: k for k, v in rel2id.items()}
        positive_predictions = [p for p in doc_result['predictions'] 
                               if p['predicted_relation'] != id2rel[0]]
        
        print(f"Found {len(positive_predictions)} vaccine-pathogen relationships:")
        
        for i, pred in enumerate(positive_predictions, 1):
            confidence = pred['probabilities'][pred['predicted_relation']]
            head_type = pred['head_entity_type']
            tail_type = pred['tail_entity_type']
            
            print(f"\n  {i}. 🔗 {pred['head_entity']} ({head_type}) → {pred['tail_entity']} ({tail_type})")
            print(f"     Confidence: {confidence:.3f}")
            # Remove input display from here
    
    return results


if __name__ == "__main__":
    results = main()

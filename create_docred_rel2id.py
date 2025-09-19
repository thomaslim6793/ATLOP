#!/usr/bin/env python3
"""
Script to create rel2id.json mapping for DocRED dataset.
Extracts all unique relation labels from train, dev, and test splits.
"""

import json
import os

def extract_relations_from_file(file_path):
    """Extract all unique relations from a DocRED format file."""
    relations = set()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for doc in data:
            if 'labels' in doc:
                for label in doc['labels']:
                    if 'r' in label:
                        relations.add(label['r'])
    
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return relations

def create_rel2id_mapping(relations):
    """Create rel2id mapping with 'N/A' as class 0."""
    # Sort relations for consistent ordering
    sorted_relations = sorted(list(relations))
    
    # Create mapping: 'N/A' gets ID 0, other relations get IDs 1, 2, 3, ...
    rel2id = {'N/A': 0}
    
    for i, rel in enumerate(sorted_relations):
        if rel != 'N/A':  # Don't duplicate N/A
            rel2id[rel] = i + 1
    
    return rel2id

def main():
    # Define paths
    data_dir = "dataset/docred"
    meta_dir = os.path.join(data_dir, "meta")
    
    # Create meta directory if it doesn't exist
    os.makedirs(meta_dir, exist_ok=True)
    
    # Extract relations from all splits
    print("Extracting relations from DocRED dataset...")
    
    train_file = os.path.join(data_dir, "train_annotated.json")
    dev_file = os.path.join(data_dir, "dev.json")
    test_file = os.path.join(data_dir, "test.json")
    
    all_relations = set()
    
    # Extract from each split
    for file_path, split_name in [(train_file, "train"), (dev_file, "dev"), (test_file, "test")]:
        relations = extract_relations_from_file(file_path)
        all_relations.update(relations)
        print(f"  {split_name}: {len(relations)} unique relations")
    
    print(f"\nTotal unique relations across all splits: {len(all_relations)}")
    
    # Create rel2id mapping
    rel2id = create_rel2id_mapping(all_relations)
    
    # Save to meta/rel2id.json
    output_path = os.path.join(meta_dir, "rel2id.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(rel2id, f, indent=2, ensure_ascii=False)
    
    print(f"\nCreated rel2id.json with {len(rel2id)} relations:")
    print(f"  Saved to: {output_path}")
    
    # Print some statistics
    print(f"\nRelation mapping preview:")
    for rel, rel_id in list(rel2id.items())[:10]:  # Show first 10
        print(f"  {rel_id}: {rel}")
    if len(rel2id) > 10:
        print(f"  ... and {len(rel2id) - 10} more relations")
    
    # Verify the mapping
    print(f"\nVerification:")
    print(f"  'N/A' mapped to ID: {rel2id.get('N/A', 'NOT FOUND')}")
    print(f"  Total relations: {len(rel2id)}")
    print(f"  Max relation ID: {max(rel2id.values())}")

if __name__ == "__main__":
    main()

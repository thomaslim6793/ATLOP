"""
Utilities to convert GliNER-style entity outputs into DocRED format.

Input per document:
- text: raw string
- entities: List[{
    'start': int,  # char start
    'end': int,    # char end (exclusive)
    'text': str,
    'label': str,  # e.g., 'vaccine', 'pathogen', 'disease'
    'score': float
  }]

Output DocRED doc:
{
  'title': str,
  'sents': List[List[str]],
  'vertexSet': List[List[{ 'name': str, 'pos': [start, end], 'sent_id': int, 'type': str }]],
  'labels': []
}
"""

from typing import List, Tuple, Dict, Optional
import re


def _tokenize_text(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    tokens = re.findall(r"\S+", text)
    token_spans: List[Tuple[int, int]] = []
    current_pos = 0
    for token in tokens:
        start_pos = text.find(token, current_pos)
        if start_pos == -1:
            start_pos = current_pos
        end_pos = start_pos + len(token)
        token_spans.append((start_pos, end_pos))
        current_pos = end_pos
    return tokens, token_spans


def _char_to_token_span(char_start: int, char_end: int, token_spans: List[Tuple[int, int]]) -> Tuple[Optional[int], Optional[int]]:
    token_start: Optional[int] = None
    token_end: Optional[int] = None
    for i, (t_start, t_end) in enumerate(token_spans):
        if char_start < t_end and char_end > t_start:
            if token_start is None:
                token_start = i
            token_end = i + 1
    return token_start, token_end


def _build_sentences_from_tokens(all_tokens: List[str], token_spans: List[Tuple[int, int]], entity_char_spans: List[Tuple[int, int]]) -> List[List[str]]:
    sentences: List[List[str]] = []
    current_idx = 0

    def overlaps_entity(span: Tuple[int, int]) -> bool:
        a, b = span
        for s, e in entity_char_spans:
            if a < e and b > s:
                return True
        return False

    while current_idx < len(all_tokens):
        end_idx = current_idx
        while end_idx < len(all_tokens):
            tok = all_tokens[end_idx]
            span = token_spans[end_idx]
            is_boundary = tok.endswith('.') or tok.endswith('!') or tok.endswith('?')
            if is_boundary and not overlaps_entity(span):
                break
            end_idx += 1
        # include end_idx token
        end_idx = min(end_idx, len(all_tokens) - 1)
        sentences.append(all_tokens[current_idx:end_idx + 1])
        current_idx = end_idx + 1
    return sentences


def _token_doc_offsets(sents: List[List[str]]) -> List[int]:
    offsets = []
    total = 0
    for sent in sents:
        offsets.append(total)
        total += len(sent)
    return offsets


def _map_token_to_sentence(token_index: int, sents: List[List[str]], offsets: List[int]) -> Tuple[int, int]:
    # returns (sent_id, sent_relative_index)
    for sid, off in enumerate(offsets):
        if token_index < off + len(sents[sid]):
            return sid, token_index - off
    # fallback to last sentence
    last_sid = len(sents) - 1
    return last_sid, max(0, token_index - offsets[last_sid])


def convert_gliner_to_docred(text: str, entities: List[Dict], title: str = "") -> Dict:
    tokens, token_spans = _tokenize_text(text)
    entity_char_spans = [(e['start'], e['end']) for e in entities]
    sents = _build_sentences_from_tokens(tokens, token_spans, entity_char_spans)
    offsets = _token_doc_offsets(sents)

    # Build mentions for vertexSet
    mentions: List[Dict] = []
    for ent in entities:
        t_start, t_end = _char_to_token_span(ent['start'], ent['end'], token_spans)
        if t_start is None or t_end is None or t_start >= t_end:
            continue
        sent_id_start, rel_start = _map_token_to_sentence(t_start, sents, offsets)
        sent_id_end, rel_end = _map_token_to_sentence(t_end - 1, sents, offsets)
        # If mention crosses sentences, clamp to the first sentence span
        if sent_id_start != sent_id_end:
            sent_id = sent_id_start
            rel_pos_start = rel_start
            rel_pos_end = len(sents[sent_id])
        else:
            sent_id = sent_id_start
            rel_pos_start = rel_start
            rel_pos_end = rel_end + 1  # exclusive end -> convert to end index

        ent_type = ent.get('label', 'UNK')
        if ent_type not in {'vaccine', 'pathogen', 'disease'}:
            ent_type = 'UNK'

        mentions.append({
            'name': ent['text'],
            'pos': [rel_pos_start, rel_pos_end],
            'sent_id': sent_id,
            'type': ent_type
        })

    # Group mentions by normalized name -> vertexSet
    groups: Dict[str, List[Dict]] = {}
    for m in mentions:
        key = m['name'].strip().lower()
        groups.setdefault(key, []).append(m)

    vertex_set = [group for _, group in groups.items() if group]

    doc = {
        'title': title,
        'sents': sents,
        'vertexSet': vertex_set,
        'labels': []
    }
    return doc


def convert_batch(texts: List[str], entities_list: List[List[Dict]], titles: Optional[List[str]] = None) -> List[Dict]:
    docs: List[Dict] = []
    if titles is None:
        titles = [""] * len(texts)
    for text, ents, title in zip(texts, entities_list, titles):
        docs.append(convert_gliner_to_docred(text, ents, title=title))
    return docs



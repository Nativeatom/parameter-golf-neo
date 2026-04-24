#!/usr/bin/env python3
"""
Tokenizer merging module for optimizing vocabulary.

Features:
1. Split tokens with multiple punctuations into separate punctuations, except `--` and `...`
2. Reduce tokens with prefix/suffix overlap by decomposition
"""

import json
import re
import os
import random
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import sentencepiece as spm
import shutil

# Import sentencepiece model proto for direct model manipulation
try:
    from sentencepiece import sentencepiece_model_pb2 as model_pb2
except ImportError:
    model_pb2 = None


def convert_freq_to_json(freq_file: str, json_file: str):
    """Convert vocabulary frequency file to JSON format."""
    tokens = []
    with open(freq_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('|')
            if len(parts) >= 5:
                try:
                    token_str = '|'.join(parts[1:-3]).strip()
                    freq_str = parts[-2].strip().replace(',', '')
                    freq = int(freq_str)
                    tokens.append({"token": token_str, "frequency": freq})
                except (ValueError, IndexError):
                    continue

    with open(json_file, 'w') as f:
        for token in tokens:
            f.write(json.dumps(token) + '\n')

    return tokens


def load_token_freq_json(json_file: str) -> List[Dict]:
    """Load token frequency data from JSON file."""
    tokens = []
    with open(json_file, 'r') as f:
        for line in f:
            token = json.loads(line.strip())
            tokens.append(token)
    return tokens


def has_leading_underline(token: str) -> bool:
    """Check if token has leading underline."""
    return token.startswith('▁')


def get_without_underline(token: str) -> str:
    """Get token without leading underline."""
    return token[1:] if token.startswith('▁') else token


def get_token_length(token: str) -> int:
    """Get length of token excluding leading underline."""
    return len(get_without_underline(token))


def find_multi_punct_tokens(tokens: List[Dict]) -> List[Dict]:
    """
    Find tokens with multiple consecutive punctuations (>=2).
    Excludes `--` and `...` from splitting.
    """
    multi_punct = []
    punct_pattern = r'[^\w\s]'

    for t in tokens:
        token_str = t['token']
        clean = get_without_underline(token_str)

        # Check for consecutive punctuation (2 or more)
        matches = list(re.finditer(punct_pattern + '{2,}', clean))

        for match in matches:
            punct_seq = match.group()
            # Skip if it's `--` or `...`
            if punct_seq in ('--', '...'):
                continue
            multi_punct.append({
                'token': token_str,
                'frequency': t['frequency'],
                'punct_seq': punct_seq,
                'match_start': match.start(),
                'match_end': match.end()
            })
            break  # Only count once per token

    return multi_punct


def split_token_by_punct(token: str) -> List[str]:
    """
    Split a token by punctuation sequences into separate tokens.
    Preserves `--` and `...` as single units.
    """
    clean = get_without_underline(token)
    has_underline = has_leading_underline(token)

    # Pattern to match: preserve `--` and `...`, split other consecutive puncts
    result = []
    i = 0
    n = len(clean)

    while i < n:
        # Check for `--` (preserve)
        if i + 1 < n and clean[i:i+2] == '--':
            if result and not result[-1].endswith('▁'):
                result.append('--')
            else:
                result.append('--')
            i += 2
        # Check for `...` (preserve)
        elif i + 2 < n and clean[i:i+3] == '...':
            if result and not result[-1].endswith('▁'):
                result.append('...')
            else:
                result.append('...')
            i += 3
        # Check for single punctuation
        elif re.match(r'[^\w\s]', clean[i]):
            # Split into individual punctuation
            if result:
                # Add as separate token
                result.append(clean[i])
            else:
                result.append(clean[i])
            i += 1
        else:
            # Regular character - accumulate
            j = i
            while j < n and not re.match(r'[^\w\s]', clean[j]):
                j += 1
            word_part = clean[i:j]
            result.append(word_part)
            i = j

    # Clean up: merge consecutive word parts and handle underlines
    final_result = []
    current_word = ""

    for part in result:
        if re.match(r'^[^\w\s]+$', part):  # Punctuation only
            if current_word:
                final_result.append('▁' + current_word if has_underline and not final_result else current_word)
                current_word = ""
            final_result.append(part)
        else:  # Word part
            current_word += part

    if current_word:
        final_result.append('▁' + current_word if has_underline and not final_result else current_word)

    # Add underline to first word part if needed
    if has_underline and final_result and not final_result[0].startswith('▁'):
        # Find first alphabetic token and add underline
        for i, part in enumerate(final_result):
            if re.search(r'[a-zA-Z0-9]', part):
                final_result[i] = '▁' + part
                break

    return final_result


def can_decompose(token: str, vocab_set: Set[str], tokenizer) -> Tuple[bool, List[str]]:
    """
    Check if a token can be decomposed into existing vocabulary tokens.
    Returns (can_decompose, parts).
    """
    clean = get_without_underline(token)
    has_underline = has_leading_underline(token)

    if not clean:
        return False, []

    # Use tokenizer to encode
    try:
        ids = tokenizer.encode(clean)
        pieces = [tokenizer.id_to_piece(id) for id in ids]

        # Check if all pieces are in vocabulary (excluding special tokens)
        valid_pieces = []
        for piece in pieces:
            if piece in vocab_set or piece.startswith('<') and piece.endswith('>'):
                valid_pieces.append(piece)
            else:
                return False, []

        # Must decompose into more than 1 piece
        if len(valid_pieces) > 1:
            # Add underline to first piece if original had it
            if has_underline and valid_pieces and not valid_pieces[0].startswith('▁'):
                # Check if first piece should have underline
                if valid_pieces[0] in vocab_set:
                    # Try to find underlined version
                    underlined = '▁' + valid_pieces[0]
                    if underlined in vocab_set:
                        valid_pieces[0] = underlined
            return True, valid_pieces

        return False, []
    except Exception:
        return False, []


def find_decomposable_tokens(
    tokens: List[Dict],
    vocab_set: Set[str],
    tokenizer,
    min_len: int,
    max_len: int
) -> List[Dict]:
    """Find tokens that can be decomposed and meet length criteria."""
    decomposable = []

    for t in tokens:
        token_str = t['token']
        token_len = get_token_length(token_str)

        if token_len < min_len or token_len > max_len:
            continue

        can_decomp, parts = can_decompose(token_str, vocab_set, tokenizer)
        if can_decomp:
            decomposable.append({
                'token': token_str,
                'frequency': t['frequency'],
                'parts': parts
            })

    return decomposable


def sample_tokens_to_remove(
    tokens: List[Dict],
    p: float,
    mode: str
) -> List[Dict]:
    """
    Sample tokens to remove based on mode.
    - uniform: randomly sample p ratio
    - inverse_freq: select least frequent p ratio
    """
    if not tokens:
        return []

    n_to_remove = int(len(tokens) * p)
    if n_to_remove == 0:
        return []

    if mode == 'uniform':
        return random.sample(tokens, min(n_to_remove, len(tokens)))
    elif mode == 'inverse_freq':
        # Sort by frequency ascending (least frequent first)
        sorted_tokens = sorted(tokens, key=lambda x: x['frequency'])
        return sorted_tokens[:n_to_remove]
    else:
        raise ValueError(f"Unknown mode: {mode}")


def get_new_vocab_size(original_size: int, tokens_to_remove: List[Dict], new_tokens: List[str]) -> int:
    """Calculate new vocabulary size after modifications."""
    removed = len(tokens_to_remove)
    added = len(new_tokens)
    return original_size - removed + added


def format_p_for_filename(p: float) -> str:
    """Format p ratio for filename (e.g., 0.1 -> 01, 0.02 -> 002)."""
    if p >= 0.1:
        return f"{int(p * 100):02d}"
    else:
        return f"{int(p * 1000):03d}"


def create_new_model(original_model_path: str, new_vocab: List[str], output_path: str) -> bool:
    """
    Create a new SentencePiece model with modified vocabulary.
    Uses ModelProto to directly manipulate the model binary.
    """
    if model_pb2 is None:
        print("Warning: sentencepiece_model_pb2 not available. Cannot create .model file.")
        return False

    try:
        # Load original model
        with open(original_model_path, 'rb') as f:
            model_proto = model_pb2.ModelProto()
            model_proto.ParseFromString(f.read())

        # Get original pieces as a dictionary for score lookup
        original_pieces = {p.piece: p for p in model_proto.pieces}

        # Create new pieces list
        new_pieces = []
        default_score = 0.0
        default_type = model_pb2.ModelProto.SentencePiece.Type.NORMAL

        for i, token in enumerate(new_vocab):
            if token in original_pieces:
                # Keep original piece with its score and type
                piece = original_pieces[token]
                new_pieces.append(piece)
            else:
                # Create new piece
                new_piece = model_proto.pieces.add()
                new_piece.piece = token
                new_piece.score = default_score
                new_piece.type = default_type
                new_pieces.append(new_piece)

        # Clear and repopulate pieces
        del model_proto.pieces[:]
        model_proto.pieces.extend(new_pieces)

        # Save new model
        with open(output_path, 'wb') as f:
            f.write(model_proto.SerializeToString())

        return True
    except Exception as e:
        print(f"Error creating model file: {e}")
        import traceback
        traceback.print_exc()
        return False


def tokenizer_merging(
    tokenizer_path: str,
    token_freq: Optional[str] = None,
    level: int = 1,
    min_token_len: int = 5,
    max_token_len: int = 20,
    p: float = 0.1,
    mode: str = 'uniform',
    output_dir: Optional[str] = None
) -> str:
    """
    Main tokenizer merging function.

    Args:
        tokenizer_path: Path to the tokenizer model file
        token_freq: Path to token frequency file (optional)
        level: 1 or 2 (Level 1: punctuation split, Level 2: + decomposition)
        min_token_len: Minimum token length (excluding underline)
        max_token_len: Maximum token length (excluding underline)
        p: Ratio of tokens to remove
        mode: 'uniform' or 'inverse_freq'
        output_dir: Output directory (default: same as tokenizer)

    Returns:
        Path to the new tokenizer file
    """
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
    original_vocab_size = tokenizer.vocab_size()
    print(f"Original vocab size: {original_vocab_size}")

    # Build vocabulary set
    vocab_set = set()
    for i in range(original_vocab_size):
        piece = tokenizer.id_to_piece(i)
        vocab_set.add(piece)

    # Load or convert token frequencies
    if token_freq:
        tokens = load_token_freq_json(token_freq)
    else:
        # Build from tokenizer
        tokens = []
        for i in range(original_vocab_size):
            piece = tokenizer.id_to_piece(i)
            tokens.append({'token': piece, 'frequency': 0})

    # Track modifications
    tokens_to_remove = set()
    new_tokens = set()

    # Level 1: Split multi-punctuation tokens
    print("Processing Level 1: Multi-punctuation tokens...")
    multi_punct = find_multi_punct_tokens(tokens)
    print(f"Found {len(multi_punct)} tokens with multi-punctuation")

    for item in multi_punct:
        tokens_to_remove.add(item['token'])
        # Split into separate tokens
        split_parts = split_token_by_punct(item['token'])
        for part in split_parts:
            if part and part not in vocab_set:
                new_tokens.add(part)

    # Level 2: Decompose tokens with overlap
    if level >= 2:
        print(f"Processing Level 2: Decomposing tokens (len {min_token_len}-{max_token_len})...")

        # Find decomposable tokens
        decomposable = find_decomposable_tokens(
            tokens, vocab_set, tokenizer,
            min_token_len, max_token_len
        )
        print(f"Found {len(decomposable)} decomposable tokens")

        # Sample tokens to remove
        to_remove = sample_tokens_to_remove(decomposable, p, mode)
        print(f"Selected {len(to_remove)} tokens to remove (p={p}, mode={mode})")

        for item in to_remove:
            tokens_to_remove.add(item['token'])
            # Add decomposition parts as new tokens
            for part in item['parts']:
                if part and part not in vocab_set:
                    new_tokens.add(part)

    # Define tokens that must always be kept (special tokens and byte pieces)
    byte_pieces = [f"<0x{i:02X}>" for i in range(256)]
    special_tokens = ['<pad>', '<s>', '</s>', '<unk>']
    must_keep = set(special_tokens + byte_pieces)

    # Calculate new vocab size
    # Account for must_keep tokens that are never removed
    actual_removed = len([t for t in tokens_to_remove if t not in must_keep])
    new_vocab_size = original_vocab_size - actual_removed + len([t for t in new_tokens if t not in must_keep and t not in vocab_set])

    # Generate output filename
    p_str = format_p_for_filename(p)
    if level >= 2:
        filename = f"fineweb_{new_vocab_size}_l{level}_min{min_token_len}_max{max_token_len}_p{p_str}_{mode}.model"
    else:
        filename = f"fineweb_{new_vocab_size}_l{level}.model"

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
    else:
        output_path = os.path.join(os.path.dirname(tokenizer_path), filename)

    print(f"New vocab size: {new_vocab_size}")
    print(f"Tokens to remove: {len(tokens_to_remove)}")
    print(f"New tokens to add: {len(new_tokens)}")

    # Create modified vocabulary file
    # Note: For a real implementation, we would need to retrain or modify the SPM model
    # This is a simplified version that creates a vocabulary file
    print(f"Creating new tokenizer at {output_path}...")

    # Build new vocabulary list
    new_vocab = []
    seen = set()

    # Keep original tokens that weren't removed (or are must_keep)
    for i in range(original_vocab_size):
        piece = tokenizer.id_to_piece(i)
        if piece in must_keep:
            new_vocab.append(piece)
            seen.add(piece)
        elif piece not in tokens_to_remove and piece not in seen:
            new_vocab.append(piece)
            seen.add(piece)

    # Add new tokens
    for token in new_tokens:
        if token not in seen:
            new_vocab.append(token)
            seen.add(token)

    # Create the new model file
    print(f"Creating new tokenizer at {output_path}...")

    success = create_new_model(tokenizer_path, new_vocab, output_path)

    if success:
        # Also create a vocabulary text file for reference
        vocab_file = output_path.replace('.model', '.vocab')
        with open(vocab_file, 'w') as f:
            for i, token in enumerate(new_vocab):
                f.write(f"{token}\t{i}\n")
        print(f"Model file saved to {output_path}")
        print(f"Vocabulary file saved to {vocab_file}")
    else:
        # Fallback to just vocab file
        vocab_file = output_path.replace('.model', '.vocab')
        with open(vocab_file, 'w') as f:
            for i, token in enumerate(new_vocab):
                f.write(f"{token}\t{i}\n")
        print(f"Vocabulary file saved to {vocab_file}")
        print(f"Note: Model proto manipulation failed, only vocab file created")

    return output_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Tokenizer Merging Tool')
    parser.add_argument('--tokenizer', required=True, help='Path to tokenizer model')
    parser.add_argument('--token-freq', help='Path to token frequency file')
    parser.add_argument('--level', type=int, default=1, choices=[1, 2], help='Merging level')
    parser.add_argument('--min-len', type=int, default=5, help='Minimum token length')
    parser.add_argument('--max-len', type=int, default=20, help='Maximum token length')
    parser.add_argument('--p', type=float, default=0.1, help='Removal ratio')
    parser.add_argument('--mode', default='uniform', choices=['uniform', 'inverse_freq'],
                        help='Sampling mode')
    parser.add_argument('--output-dir', help='Output directory')

    args = parser.parse_args()

    tokenizer_merging(
        tokenizer_path=args.tokenizer,
        token_freq=args.token_freq,
        level=args.level,
        min_token_len=args.min_len,
        max_token_len=args.max_len,
        p=args.p,
        mode=args.mode,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()

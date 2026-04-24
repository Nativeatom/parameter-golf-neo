"""
Dataset Normalization Pipeline

Normalizes FineWeb binary datasets by:
1. Decoding tokens to text
2. Applying text normalization (URLs, emails, punctuation, etc.)
3. Re-encoding to tokens
4. Writing normalized binary files

Usage:
    python normalize_dataset.py \
        data/datasets/fineweb10B_sp8192 \
        train \
        --output data/datasets/fineweb10B_sp8192_clean_v1 \
        --tokenizer data/tokenizers/fineweb_8192_bpe.model \
        --vocab-size 8192
"""

import argparse
import glob
import multiprocessing as mp
import os
import struct
import sys
from functools import partial
from pathlib import Path

import numpy as np
import sentencepiece as spm

# Add the current directory to path to import normalization
sys.path.insert(0, str(Path(__file__).parent))
from normalization import normalize_document


def init_worker(tokenizer_path, vocab_size, seq_len):
    """Initialize worker process with tokenizer."""
    global _tokenizer, _vocab_size, _seq_len
    _tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
    _vocab_size = vocab_size
    _seq_len = seq_len


def load_fineweb_bin(bin_path: str, tokenizer=None, seq_len: int = 8192):
    """
    Load documents from FineWeb binary format.

    Returns:
        List of (doc_id, text) tuples
    """
    # Use global tokenizer in worker processes
    if tokenizer is None:
        tokenizer = _tokenizer
        seq_len = _seq_len

    vocab_size = int(tokenizer.vocab_size())

    # Read header (256 int32 values)
    header = np.fromfile(bin_path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {bin_path}")
    num_tokens = int(header[2])

    # Read tokens (uint16, after header)
    header_bytes = 256 * np.dtype("<i4").itemsize  # 1024 bytes
    tokens = np.fromfile(bin_path, dtype="<u2", count=num_tokens, offset=header_bytes)

    # Split into documents (each seq_len tokens)
    num_docs = len(tokens) // seq_len

    documents = []
    for i in range(num_docs):
        doc_tokens = tokens[i * seq_len:(i + 1) * seq_len]
        # Filter to valid token range (0 to vocab_size-1)
        valid_tokens = [int(t) for t in doc_tokens if 0 < t < vocab_size]
        # Decode to text
        text = tokenizer.DecodeIds(valid_tokens)
        documents.append((i, text))

    return documents, num_tokens


def encode_text_to_tokens(text: str, tokenizer=None, vocab_size: int = None, seq_len: int = None):
    """
    Encode text back to token IDs and pad/truncate to seq_len.

    Returns:
        numpy array of shape (seq_len,) with token IDs
    """
    # Use global tokenizer in worker processes
    if tokenizer is None:
        tokenizer = _tokenizer
        vocab_size = _vocab_size
        seq_len = _seq_len

    # Encode text to token IDs
    token_ids = tokenizer.EncodeAsIds(text)

    # Filter to valid range
    token_ids = [t for t in token_ids if 0 < t < vocab_size]

    # Create output array
    output = np.zeros(seq_len, dtype=np.uint16)

    # Copy tokens (truncate if too long)
    num_tokens = min(len(token_ids), seq_len)
    output[:num_tokens] = token_ids[:num_tokens]

    return output


def write_bin_file(output_path: str, all_tokens: np.ndarray, num_docs: int):
    """
    Write documents to FineWeb binary format.

    Format:
    - 256 int32 header (1024 bytes)
      - header[0] = magic number (20240520)
      - header[1] = version (1)
      - header[2] = number of tokens in file
    - Remaining: uint16 tokens
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create header
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520  # magic number
    header[1] = 1  # version
    header[2] = len(all_tokens)  # total tokens

    # Write file
    with open(output_path, 'wb') as f:
        f.write(header.tobytes())
        f.write(all_tokens.tobytes())

    print(f"  Wrote {num_docs} documents ({len(all_tokens)} tokens) to {output_path}")


def process_shard_worker(args):
    """
    Worker function to process a single shard file in parallel.
    Args is a tuple of (input_path, output_path).
    """
    input_path, output_path = args
    return process_shard(input_path, output_path, None, None, None)


def process_shard(input_path: str, output_path: str, tokenizer, vocab_size: int, seq_len: int = 8192):
    """
    Process a single shard file: load, normalize, and write.
    """
    # Use global values in worker processes
    if tokenizer is None:
        tokenizer = _tokenizer
        vocab_size = _vocab_size
        seq_len = _seq_len

    print(f"\nProcessing {input_path}...")

    # Load documents
    documents, original_num_tokens = load_fineweb_bin(input_path, tokenizer, seq_len)
    print(f"  Loaded {len(documents)} documents ({original_num_tokens} tokens)")

    # Normalize each document
    normalized_docs = []
    for doc_id, text in documents:
        normalized_text = normalize_document(text)
        normalized_docs.append(normalized_text)

    # Re-encode to tokens
    all_tokens = []
    for text in normalized_docs:
        tokens = encode_text_to_tokens(text, tokenizer, vocab_size, seq_len)
        all_tokens.append(tokens)

    # Concatenate all tokens
    all_tokens = np.concatenate(all_tokens)

    # Write output
    write_bin_file(output_path, all_tokens, len(normalized_docs))
    return input_path


def main():
    parser = argparse.ArgumentParser(description="Normalize FineWeb dataset")
    parser.add_argument("--dataset_dir", type=str,
                        help="Path to the dataset directory containing .bin files")
    parser.add_argument("--split", type=str, choices=["train", "val"],
                        help="Dataset split to process: 'train' or 'val'")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for normalized files")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Path to SentencePiece tokenizer model")
    parser.add_argument("--vocab-size", type=int, default=8192,
                        help="Vocabulary size")
    parser.add_argument("--seq-len", type=int, default=8192,
                        help="Sequence length (tokens per document)")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of parallel workers (default: CPU count - 1)")

    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer}...")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    print(f"  Loaded tokenizer with vocab size {args.vocab_size}")

    # Build input file pattern based on split
    input_pattern = os.path.join(args.dataset_dir, f"fineweb_{args.split}_*.bin")

    # Find input files
    input_files = sorted(glob.glob(input_pattern))
    if not input_files:
        print(f"No files found matching pattern: {input_pattern}")
        sys.exit(1)

    print(f"\nFound {len(input_files)} input files for '{args.split}' split")

    # Prepare arguments for parallel processing
    shard_args = []
    for input_path in input_files:
        input_filename = os.path.basename(input_path)
        output_path = os.path.join(args.output, input_filename)
        shard_args.append((input_path, output_path))

    # Process shards in parallel using multiprocessing
    num_workers = args.num_workers if args.num_workers else max(1, mp.cpu_count() - 1)
    print(f"Processing with {num_workers} workers...")

    with mp.Pool(processes=num_workers, initializer=init_worker,
                 initargs=(args.tokenizer, args.vocab_size, args.seq_len)) as pool:
        pool.map(process_shard_worker, shard_args)

    print("\nNormalization complete!")


if __name__ == "__main__":
    main()

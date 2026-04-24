"""
MinHash-based Near-Duplicate Detection for Deep Learning Datasets

This module implements Locality Sensitive Hashing (LSH) with MinHash to efficiently
find and remove near-duplicate documents from large text datasets.

================================================================================
INTUITION FOR DEEP LEARNING PRACTITIONERS
================================================================================

Think of MinHash like a "soft fingerprint" for documents:
- Traditional dedup: exact string matching (lossy - misses near-duplicates)
- MinHash dedup:     Jaccard similarity of n-grams (finds near-duplicates)

Why this matters for training LLMs:
1. Duplicate data causes overfitting (model memorizes instead of generalizes)
2. Near-duplicates (e.g., copy-pasted text with small changes) are common in web data
3. Removing them improves perplexity and downstream task performance

================================================================================
HOW MINHASH WORKS (THE QUICK VERSION)
================================================================================

1. SHINGLING: Convert document into overlapping n-grams (3-grams = "abc", "bcd", "cde")
   Like a sliding window over tokenized text

2. HASHING: Apply many random hash functions to each shingle
   Like projecting data into many random directions (similar to random features)

3. MIN-SELECTION: For each hash function, keep the MINIMUM hash value
   The "min" in MinHash - creates a compact signature

4. LSH BANDING: Split signature into bands, hash each band to buckets
   Documents with any bucket collision are "candidates" for similarity check
   This is the key speedup: O(n) instead of O(n²) comparisons

5. VERIFICATION: Compute exact Jaccard similarity for candidates only
   Remove documents above threshold (keeping higher quality score)

================================================================================
TIME COMPLEXITY
================================================================================

For N documents with average length L:
- Shingling + MinHash:    O(N × L)      - single pass, parallelizable
- LSH bucketing:          O(N × B)      - B = number of bands (~20-50)
- Candidate verification: O(C × K)      - C = candidates, K = signature length

Typical speed: ~1,000-10,000 docs/sec on single CPU core
Memory: O(N × K × 4 bytes) for signatures (~800 bytes per doc with K=200)

================================================================================
"""

import hashlib
import json
import os
import struct
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Set, Tuple
import sentencepiece as spm

import numpy as np
from datasketch import MinHash, MinHashLSH
from transformers import AutoTokenizer
import pdb

# ==============================================================================
# N-GRAM PROCESSING
# ==============================================================================

class Ngram:
    """
    Convert token sequence into overlapping n-grams (shingles).

    Example with n=3:
        Input:  ["the", "cat", "sat", "on", "mat"]
        Output: ["the|cat|sat", "cat|sat|on", "sat|on|mat"]

    Why n-grams? They capture local word order better than bag-of-words.
    n=3 is a good default (captures short phrases without being too sparse).
    """

    def __init__(self, n: int = 3, separator: str = "|"):
        self.n = n
        self.sep = separator

    def __call__(self, tokens: List[str]) -> List[str]:
        """Generate n-grams from token list."""
        if len(tokens) < self.n:
            return [self.sep.join(tokens)]
        return [self.sep.join(tokens[i:i + self.n]) for i in range(len(tokens) - self.n + 1)]


# ==============================================================================
# HASH UTILITIES
# ==============================================================================

def md5_hash(s: str) -> str:
    """Compute MD5 hash of string (for LSH bucketing)."""
    return hashlib.md5(s.encode()).hexdigest()


def compute_minhash(
    text: str,
    tokenizer,
    ngram: Ngram,
    num_perm: int = 200
) -> np.ndarray:
    """
    Compute MinHash signature for a text document.

    Args:
        text: Raw text string
        tokenizer: HuggingFace tokenizer or SentencePieceProcessor
        ngram: Ngram generator (typically 3-grams)
        num_perm: Number of random permutations (hash functions)
                  Higher = more accurate but slower and more memory
                  128-256 is typical range; 200 is a good default

    Returns:
        numpy array of shape (num_perm,) containing the MinHash signature
    """
    # Get token strings - handle both HF tokenizers and sentencepiece
    if hasattr(tokenizer, 'tokenize') and callable(getattr(tokenizer, 'tokenize', None)):
        try:
            # HuggingFace tokenizer returns strings
            tokens = tokenizer.tokenize(text)
            if tokens and isinstance(tokens[0], str):
                pass  # HF tokenizer
            else:
                raise ValueError("Got non-string tokens")
        except:
            # SentencePiece style
            tokens = tokenizer.encode_as_pieces(text)
    else:
        # SentencePiece
        tokens = tokenizer.encode_as_pieces(text)

    shingles = ngram(tokens)

    m = MinHash(num_perm=num_perm)
    # Update with encoded shingles (datasketch expects bytes)
    m.update_batch([shingle.encode('utf-8') for shingle in shingles])

    return m.digest()


def get_lsh_buckets(minhash: np.ndarray, num_bands: int) -> List[str]:
    """
    Generate LSH bucket keys from MinHash signature.

    The signature is split into `num_bands` rows each. Each band is hashed
    to a bucket key. Documents sharing ANY bucket key are similarity candidates.

    Key insight: P(collision) = 1 - (1 - J^r)^b where:
        - J = Jaccard similarity
        - r = rows per band
        - b = number of bands

    This creates an S-curve: similar docs likely collide, dissimilar ones unlikely.

    Args:
        minhash: MinHash signature array
        num_bands: Number of bands (typically 20-50)

    Returns:
        List of bucket keys (strings)
    """
    num_perm = len(minhash)
    rows_per_band = num_perm // num_bands
    buckets = []

    for band_idx in range(num_bands):
        start = band_idx * rows_per_band
        end = (band_idx + 1) * rows_per_band
        band_values = minhash[start:end]
        # Create deterministic bucket key from band values
        band_str = ','.join([str(int(v)) for v in band_values])
        bucket_key = f"band:{band_idx}:{md5_hash(band_str)}"
        buckets.append(bucket_key)

    return buckets


def jaccard_similarity(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """
    Compute Jaccard similarity between two MinHash signatures.

    Jaccard(A, B) = |A ∩ B| / |A ∪ B|

    For MinHash signatures, this is approximated by the fraction of
    matching hash values (since MinHash is a random projection that
    preserves Jaccard similarity in expectation).
    """
    assert len(sig1) == len(sig2), "Signatures must have same length"
    matches = np.sum(sig1 == sig2)
    return float(matches) / len(sig1)


# ==============================================================================
# DOCUMENT PROCESSING
# ==============================================================================

def process_document(
    doc_id: int,
    text: str,
    quality_score: float,
    tokenizer: AutoTokenizer,
    ngram: Ngram,
    num_perm: int = 200,
    num_bands: int = 20
) -> Dict:
    """
    Process a single document: tokenize, compute MinHash, get LSH buckets.

    Returns dict with all metadata needed for deduplication.
    """
    minhash = compute_minhash(text, tokenizer, ngram, num_perm)
    buckets = get_lsh_buckets(minhash, num_bands)

    return {
        'doc_id': doc_id,
        'text': text,
        'quality_score': quality_score,
        'minhash': minhash,
        'lsh_buckets': buckets,
    }


def group_by_buckets(documents: List[Dict]) -> Dict[str, List[int]]:
    """
    Build bucket -> document indices mapping.

    This is the core of LSH: documents in the same bucket are candidates.
    """
    bucket_to_docs = defaultdict(list)
    for doc in documents:
        for bucket in doc['lsh_buckets']:
            bucket_to_docs[bucket].append(doc['doc_id'])
    return bucket_to_docs


def find_candidate_pairs(bucket_to_docs: Dict[str, List[int]]) -> Set[Tuple[int, int]]:
    """
    Find all candidate document pairs that share at least one LSH bucket.

    These are the only pairs we need to check for actual similarity.
    """
    candidates = set()
    for doc_ids in bucket_to_docs.values():
        if len(doc_ids) < 2:
            continue
        # Add all pairs from this bucket
        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):
                # Ensure consistent ordering (smaller id first)
                pair = (min(doc_ids[i], doc_ids[j]), max(doc_ids[i], doc_ids[j]))
                candidates.add(pair)
    return candidates


def deduplicate_documents(
    documents: List[Dict],
    similarity_threshold: float = 0.8,
    verbose: bool = True
) -> Tuple[Set[int], Dict[int, int]]:
    """
    Remove near-duplicate documents using MinHash + LSH.

    Strategy: Keep the document with highest quality_score from each duplicate cluster.
    This is like "max pooling" over duplicate groups.

    Args:
        documents: List of processed document dicts (from process_document)
        similarity_threshold: Jaccard similarity threshold (0.8 = 80% similar)
        verbose: Print progress information

    Returns:
        (keep_ids, duplicate_map) where:
            - keep_ids: set of document IDs to keep
            - duplicate_map: dict mapping removed_id -> kept_id
    """
    start_time = time.time()

    # Build LSH index
    if verbose:
        print(f"Building LSH index for {len(documents)} documents...")
    bucket_to_docs = group_by_buckets(documents)

    if verbose:
        print(f"  Created {len(bucket_to_docs)} buckets")

    # Find candidates
    candidates = find_candidate_pairs(bucket_to_docs)
    if verbose:
        print(f"  Found {len(candidates)} candidate pairs")
        total_pairs = len(documents) * (len(documents) - 1) // 2
        print(f"  Pruned {total_pairs - len(candidates)} / {total_pairs} pairs ({100*(1 - len(candidates)/max(total_pairs,1)):.1f}%)")

    # Build document lookup
    doc_by_id = {doc['doc_id']: doc for doc in documents}

    # Sort documents by quality score (descending) - keep highest quality
    sorted_docs = sorted(documents, key=lambda x: x['quality_score'], reverse=True)

    # Track which documents to keep
    keep_ids = set()
    duplicate_map = {}  # removed_id -> kept_id

    # Build adjacency list from candidates for faster lookup
    doc_to_candidates = defaultdict(set)
    for (id1, id2) in candidates:
        doc_to_candidates[id1].add(id2)
        doc_to_candidates[id2].add(id1)

    if verbose:
        print(f"\nVerifying candidates (threshold={similarity_threshold})...")

    max_sims = []

    # Greedy clustering: process in quality order
    for doc in sorted_docs:
        doc_id = doc['doc_id']
        if doc_id in keep_ids or doc_id in duplicate_map:
            continue

        # Keep this document (highest quality in its cluster)
        keep_ids.add(doc_id)

        max_sim = 0

        # Find all similar documents from candidates
        for other_id in doc_to_candidates[doc_id]:
            if other_id in keep_ids or other_id in duplicate_map:
                continue

            other_doc = doc_by_id[other_id]
            sim = jaccard_similarity(doc['minhash'], other_doc['minhash'])
            max_sim = max(sim, max_sim)
            if sim >= similarity_threshold:
                duplicate_map[other_id] = doc_id
        
        max_sims.append({"doc_id": doc_id, "max_sim": max_sim, "text": doc["text"]})


    max_sim_scores_all = [x["max_sim"] for x in max_sims]
    max_sim_of_all = max(max_sim_scores_all)
    mean_sim_of_all = np.mean(max_sim_scores_all)
    above_half_sim_of_all = len([s for s in max_sim_scores_all if s >= 0.5]) / len(max_sim_scores_all)
    zero_sim_of_all = len([s for s in max_sim_scores_all if s == 0]) / len(max_sim_scores_all)

    print("max_sim_score={} mean_sim_score={} above_half_sim_ratio={}".format(max_sim_of_all, round(mean_sim_of_all, 5), round(above_half_sim_of_all, 5), round(zero_sim_of_all, 5)))


    elapsed = time.time() - start_time
    if verbose:
        print(f"\nDeduplication complete in {elapsed:.2f}s")
        print(f"  Kept: {len(keep_ids)} documents")
        print(f"  Removed: {len(duplicate_map)} duplicates ({100*len(duplicate_map)/len(documents):.1f}%)")

    pdb.set_trace()

    return keep_ids, duplicate_map


# ==============================================================================
# DATA LOADING (FOR .bin FILES)
# ==============================================================================

def load_fineweb_bin(
    bin_path: str,
    tokenizer,
    max_docs: Optional[int] = None,
    seq_len: int = 8192
) -> Iterator[Tuple[int, str]]:
    """
    Load documents from FineWeb binary format.

    FineWeb .bin format (from train_gpt.py):
    - 256 int32 header (1024 bytes)
      - header[0] = magic number (20240520)
      - header[1] = version (1)
      - header[2] = number of tokens in file
    - Remaining: uint16 tokens

    Args:
        bin_path: Path to .bin file
        tokenizer: Tokenizer for decoding (AutoTokenizer or SentencePieceProcessor)
        max_docs: Maximum number of documents to load (None = all)
        seq_len: Sequence length (tokens per document)

    Yields:
        (doc_id, text) tuples
    """
    # Get vocab size (handle both HF and sentencepiece tokenizers)
    if hasattr(tokenizer, 'vocab_size'):
        vocab_size = tokenizer.vocab_size
        if callable(vocab_size):
            vocab_size = vocab_size()
    else:
        vocab_size = tokenizer.vocab_size()

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
    if max_docs:
        num_docs = min(num_docs, max_docs)

    for i in range(num_docs):
        doc_tokens = tokens[i * seq_len:(i + 1) * seq_len]
        # Filter to valid token range (0 to vocab_size-1)
        valid_tokens = [int(t) for t in doc_tokens if 0 < t < vocab_size]
        # Decode to text
        if hasattr(tokenizer, 'decode'):
            # HuggingFace tokenizer
            try:
                text = tokenizer.decode(valid_tokens, skip_special_tokens=True)
            except TypeError:
                # SentencePiece wrapper fallback
                text = tokenizer.DecodeIds(valid_tokens)
        else:
            # Raw sentencepiece
            text = tokenizer.DecodeIds(valid_tokens)
        yield i, text


def estimate_quality(text: str) -> float:
    """
    Estimate document quality for deduplication priority.

    Higher score = keep this document over its near-duplicates.
    Simple heuristic: longer documents with more unique words are better.

    In practice, you might use:
    - Perplexity from a small language model
    - Quality classifier scores
    - Source domain reputation
    """
    words = text.split()
    if not words:
        return 0.0

    # Length score (log-scaled to avoid excessive bias toward very long docs)
    length_score = min(len(words) / 100.0, 10.0)

    # Diversity score (unique words / total words)
    unique_ratio = len(set(words)) / len(words)

    # Combine
    return length_score + unique_ratio * 5.0


# ==============================================================================
# MAIN DEDUPLICATION PIPELINE
# ==============================================================================

def deduplicate_dataset(
    data_path: str,
    tokenizer_path: str,
    output_path: Optional[str] = None,
    num_perm: int = 200,
    num_bands: int = 20,
    similarity_threshold: float = 0.8,
    max_docs: Optional[int] = None,
    seq_len: int = 8192,
    vocab_size: int=8192,
    verbose: bool = True
) -> Dict:
    """
    End-to-end MinHash deduplication pipeline for FineWeb-style datasets.

    This is the main entry point - call this function with paths to get
    deduplicated document IDs.

    ==============================================================================
    EXAMPLE USAGE
    ==============================================================================

    ```python
    from utils import deduplicate_dataset

    results = deduplicate_dataset(
        data_path="./data/datasets/fineweb10B_sp8192/fineweb_train_000000.bin",
        tokenizer_path="./data/tokenizers/fineweb_8192_bpe.model",
        num_perm=200,          # Signature size (200*4=800 bytes per doc)
        num_bands=20,          # LSH bands (higher = more candidates)
        similarity_threshold=0.8,  # Jaccard threshold for duplicates
        max_docs=10000,        # Process only first 10k docs for testing
    )

    print(f"Kept {len(results['keep_ids'])} documents")
    print(f"Removed {len(results['duplicates'])} duplicates")
    ```

    ==============================================================================
    TIME ESTIMATES (APPROXIMATE, SINGLE CPU CORE)
    ==============================================================================

    | Documents | Time (200 perm) | Time (128 perm) | Memory  |
    |-----------|-----------------|-----------------|---------|
    | 1,000     | 5-10 sec        | 3-5 sec         | 1 MB    |
    | 10,000    | 1-2 min         | 30-60 sec       | 8 MB    |
    | 100,000   | 10-20 min       | 5-10 min        | 80 MB   |
    | 1,000,000 | 2-4 hours       | 1-2 hours       | 800 MB  |

    For large-scale processing, parallelize document processing across cores.

    ==============================================================================
    PARAMETER GUIDE
    ==============================================================================

    - num_perm (signature size):
        - 128: Fast, less accurate (good for large-scale filtering)
        - 200: Balanced (recommended default)
        - 256: Slower, more accurate (for high-precision requirements)

    - num_bands (LSH parameter):
        - Controls candidate pair recall
        - Rule of thumb: num_perm / num_bands should be ~10
        - More bands = more candidates = slower but better recall

    - similarity_threshold:
        - 0.9: Very strict (only near-identical documents)
        - 0.8: Standard (recommended, catches rewrites with small changes)
        - 0.7: Lenient (catches loosely related documents)

    Returns:
        Dictionary with:
            - 'keep_ids': Set of document IDs to keep
            - 'duplicates': Dict mapping removed_id -> kept_id
            - 'stats': Processing statistics
            - 'config': Parameters used
    """
    overall_start = time.time()

    # Initialize
    if verbose:
        print("=" * 70)
        print("MinHash Deduplication Pipeline")
        print("=" * 70)
        print(f"Data: {data_path}")
        print(f"Tokenizer: {tokenizer_path}")
        print(f"Parameters: {num_perm} permutations, {num_bands} bands, threshold={similarity_threshold}")
        print()

    # Load tokenizer
    tokenizer_load_start = time.time()

    # Try different tokenizer formats
    if os.path.exists(tokenizer_path):
        sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
        if int(sp.vocab_size()) != vocab_size:
            raise ValueError(
                f"VOCAB_SIZE={vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
            )
        # self.val_tokens = load_validation_tokens(h.val_files, h.eval_seq_len)
        # self.base_bytes_lut, self.has_leading_space_lut, self.is_boundary_token_lut = (
        #     build_sentencepiece_luts(self.sp, h.vocab_size, device))

        # Wrap in HF-like interface
        tokenizer = sp
        tokenizer.tokenize = sp.encode_as_pieces
        tokenizer.decode = sp.decode_ids
        tokenizer.vocab_size = sp.vocab_size()
    

    if verbose:
        print(f"  Loaded in {time.time() - tokenizer_load_start:.2f}s")
        print(f"  Vocab size: {tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else tokenizer.vocab_size()}\n")

    # Initialize n-gram generator
    ngram = Ngram(n=3)

    # Load and process documents
    if verbose:
        print("Loading and processing documents...")
    doc_process_start = time.time()

    documents = []
    for doc_id, text in load_fineweb_bin(data_path, tokenizer, max_docs, seq_len):
        quality = estimate_quality(text)
        doc = process_document(
            doc_id=doc_id,
            text=text,
            quality_score=quality,
            tokenizer=tokenizer,
            ngram=ngram,
            num_perm=num_perm,
            num_bands=num_bands
        )
        documents.append(doc)

        if verbose and (doc_id + 1) % 1000 == 0:
            print(f"  Processed {doc_id + 1} documents...")

    doc_process_time = time.time() - doc_process_start
    if verbose:
        print(f"  Loaded {len(documents)} documents in {doc_process_time:.2f}s")
        print(f"  Speed: {len(documents)/doc_process_time:.1f} docs/sec")
        print()

    # Deduplication
    keep_ids, duplicate_map = deduplicate_documents(
        documents,
        similarity_threshold=similarity_threshold,
        verbose=verbose
    )

    # Compile results
    overall_time = time.time() - overall_start

    results = {
        'keep_ids': keep_ids,
        'duplicates': duplicate_map,
        'stats': {
            'total_docs': len(documents),
            'kept_docs': len(keep_ids),
            'removed_docs': len(duplicate_map),
            'dedup_ratio': len(duplicate_map) / len(documents) if documents else 0,
            'processing_time_sec': overall_time,
            'docs_per_sec': len(documents) / overall_time if overall_time > 0 else 0,
        },
        'config': {
            'num_perm': num_perm,
            'num_bands': num_bands,
            'similarity_threshold': similarity_threshold,
            'data_path': data_path,
        }
    }

    # Save results if output path provided
    if output_path:
        if verbose:
            print(f"\nSaving results to {output_path}...")
        with open(output_path, 'w') as f:
            # Convert sets to lists for JSON serialization
            save_data = {
                'keep_ids': sorted(list(keep_ids)),
                'duplicates': duplicate_map,
                'stats': results['stats'],
                'config': results['config'],
            }
            json.dump(save_data, f, indent=2)

    if verbose:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Total documents:    {results['stats']['total_docs']}")
        print(f"Kept documents:     {results['stats']['kept_docs']}")
        print(f"Removed duplicates: {results['stats']['removed_docs']}")
        print(f"Dedup ratio:        {results['stats']['dedup_ratio']:.1%}")
        print(f"Total time:         {results['stats']['processing_time_sec']:.1f}s")
        print(f"Throughput:         {results['stats']['docs_per_sec']:.1f} docs/sec")
        print("=" * 70)

    return results


# ==============================================================================
# TEST / DEMO FUNCTION
# ==============================================================================

def test_deduplication():
    """
    Quick test to verify near-duplicate detection works correctly.
    Creates synthetic documents with known near-duplicates.
    """
    print("=" * 70)
    print("Testing MinHash Deduplication with Synthetic Data")
    print("=" * 70)

    # Create synthetic documents with CLEAR near-duplicate clusters
    documents = []

    # Cluster 1: Coffee articles (docs 0-2 are near-duplicates, doc 3 is different within cluster)
    base_coffee = "Coffee is a brewed drink prepared from roasted coffee beans. " * 10
    for i in range(3):
        # Near-duplicates: ~90% similar
        modified = base_coffee.replace("roasted", "processed").replace("brewed", "prepared") + f" Extra {i}."
        doc = {
            'doc_id': i,
            'text': modified,
            'quality_score': 10.0 - i,  # Doc 0 has highest quality
            'minhash': None,
            'lsh_buckets': [],
        }
        documents.append(doc)

    # Cluster 2: Tea articles (docs 3-5 are near-duplicates)
    base_tea = "Tea is an aromatic beverage prepared by pouring hot water over tea leaves. " * 10
    for i in range(3):
        modified = base_tea.replace("hot", "warm").replace("leaves", "leaf") + f" Variation {i}."
        doc = {
            'doc_id': i + 3,
            'text': modified,
            'quality_score': 8.0 - i,
            'minhash': None,
            'lsh_buckets': [],
        }
        documents.append(doc)

    # Cluster 3: ML articles (docs 6-7 are near-duplicates)
    base_ml = "Machine learning is a method of data analysis that automates analytical model building. " * 10
    for i in range(2):
        modified = base_ml.replace("automates", "enables").replace("method", "approach") + f" Ver {i}."
        doc = {
            'doc_id': i + 6,
            'text': modified,
            'quality_score': 6.0 - i,
            'minhash': None,
            'lsh_buckets': [],
        }
        documents.append(doc)

    # Unique documents (completely different topics - should NOT be duplicates)
    unique_docs = [
        "Basketball is a team sport played on a rectangular court. " * 10,
        "Photosynthesis is the process by which plants use sunlight to synthesize foods. " * 10,
        "Quantum mechanics is a fundamental theory in physics describing nature at small scales. " * 10,
    ]
    for i, text in enumerate(unique_docs):
        doc = {
            'doc_id': i + 8,
            'text': text,
            'quality_score': 5.0,
            'minhash': None,
            'lsh_buckets': [],
        }
        documents.append(doc)

    print(f"\nCreated {len(documents)} synthetic documents in 3 clusters + 3 unique:")
    print(f"  - Cluster 1 (Coffee): docs 0,1,2 are ~90% similar")
    print(f"  - Cluster 2 (Tea): docs 3,4,5 are ~90% similar")
    print(f"  - Cluster 3 (ML): docs 6,7 are ~90% similar")
    print(f"  - Unique: docs 8,9,10 are completely different")

    # Compute MinHash for all documents using simple word n-grams
    ngram = Ngram(n=3, separator=" ")
    for doc in documents:
        tokens = doc['text'].lower().split()
        shingles = ngram(tokens)
        m = MinHash(num_perm=128)
        m.update_batch([s.encode('utf-8') for s in shingles])
        doc['minhash'] = m.digest()
        doc['lsh_buckets'] = get_lsh_buckets(doc['minhash'], num_bands=16)

    # Show actual similarities within clusters
    print("\nActual similarities (ground truth):")
    for i, j in [(0, 1), (0, 2), (3, 4), (6, 7), (0, 3), (0, 8)]:
        sim = jaccard_similarity(documents[i]['minhash'], documents[j]['minhash'])
        print(f"  Doc {i} vs Doc {j}: {sim:.3f}")

    # Run deduplication
    keep_ids, duplicate_map = deduplicate_documents(
        documents,
        similarity_threshold=0.7,  # Should catch the ~90% similar docs
        verbose=True
    )

    # Verify results
    print("\n" + "=" * 70)
    print("Verification (expected behavior)")
    print("=" * 70)

    # Expected: keep highest quality from each cluster
    expected_kept = {0, 3, 6, 8, 9, 10}  # Highest quality in each group + all unique
    expected_removed = {1: 0, 2: 0, 4: 3, 5: 3, 7: 6}  # Lower quality marked as dups

    correct = 0
    total = 0

    # Check kept documents
    for doc_id in expected_kept:
        total += 1
        if doc_id in keep_ids:
            print(f"  Doc {doc_id}: correctly kept")
            correct += 1
        else:
            print(f"  Doc {doc_id}: ERROR - should be kept but was removed (dup of {duplicate_map.get(doc_id)})")

    # Check removed documents
    for dup_id, orig_id in expected_removed.items():
        total += 1
        if dup_id in duplicate_map:
            if duplicate_map[dup_id] == orig_id:
                print(f"  Doc {dup_id}: correctly marked as duplicate of {orig_id}")
                correct += 1
            else:
                print(f"  Doc {dup_id}: marked as dup of {duplicate_map[dup_id]} (expected {orig_id})")
        else:
            print(f"  Doc {dup_id}: ERROR - should be marked as duplicate of {orig_id}")

    print(f"\nTest accuracy: {correct}/{total} checks passed")
    return correct == total


# ==============================================================================
# MINIMAL RUNNABLE EXAMPLE
# ==============================================================================

if __name__ == "__main__":
    """
    Run this file directly to test deduplication on a single FineWeb shard.

    Example:
        python utils.py

    This will process the first 5000 documents from fineweb10B_sp8192
    and output deduplication statistics.
    """
    import argparse

    parser = argparse.ArgumentParser(description="MinHash deduplication for FineWeb")
    # Default paths are relative to this file's location
    # utils.py is in: ./records/track_10min_16mb/2026-04-05_PR1394_adapted/
    # Data is in: ./data/ (relative to project root)
    # Project root is 3 levels up: records/ -> track_10min_16mb/ -> 2026-04-05_PR1394_adapted/ -> utils.py
    _file_dir = Path(__file__).parent
    _project_root = _file_dir.parent.parent.parent  # Go up to project root (parameter-golf-neo)
    _default_data = _project_root / "data" / "datasets" / "fineweb10B_sp1024" / "fineweb_train_000000.bin"
    _default_tokenizer = _project_root / "data" / "tokenizers"

    parser.add_argument("--data-path", type=str,
                        default=str(_default_data),
                        help="Path to .bin file")
    parser.add_argument("--tokenizer-path", type=str,
                        default=str(_default_tokenizer),
                        help="Path to tokenizer")
    parser.add_argument("--max-docs", type=int, default=5000,
                        help="Maximum documents to process (for testing)")
    parser.add_argument("--num-perm", type=int, default=256,
                        help="Number of MinHash permutations (128=faster, 200=accurate)")
    parser.add_argument("--num-bands", type=int, default=16,
                        help="Number of LSH bands")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Jaccard similarity threshold for duplicates")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")
    parser.add_argument("--vocab_size", type=int, default=8192,
                        help="The vocab size of the tokenizer")
    parser.add_argument("--seq-len", type=int, default=1024,
                        help="Sequence length (tokens per document)")
    parser.add_argument("--test", action="store_true",
                        help="Run synthetic test to verify deduplication works")

    args = parser.parse_args()

    # Run synthetic test if requested
    if args.test:
        success = test_deduplication()
        exit(0 if success else 1)

    # Check if files exist, provide helpful message if not
    if not os.path.exists(args.data_path):
        print(f"ERROR: Data file not found: {args.data_path}")
        print("\nTo download the dataset, run:")
        print("  python data/cached_challenge_fineweb.py --variant sp8192 --train-shards 1")
        print("\nOr use a different data path.")
        exit(1)

    tokenizer_path = os.path.join(args.tokenizer_path, f'fineweb_{args.vocab_size}_bpe.model')

    # Run deduplication
    results = deduplicate_dataset(
        data_path=args.data_path,
        tokenizer_path=tokenizer_path,
        vocab_size=args.vocab_size,
        output_path=args.output,
        num_perm=args.num_perm,
        num_bands=args.num_bands,
        similarity_threshold=args.threshold,
        max_docs=args.max_docs,
        seq_len=args.seq_len,
        verbose=True
    )
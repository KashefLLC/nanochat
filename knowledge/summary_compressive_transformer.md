# Compressive Transformers for Long-Range Sequence Modelling

**Paper:** Rae, Potapenko, Jayakumar, Hillier, Lillicrap (DeepMind, 2019)
**Venue:** ICLR 2020
**arXiv:** [1911.05507](https://arxiv.org/abs/1911.05507)

---

## Core Idea

The Compressive Transformer extends the TransformerXL by **compressing old memories instead of discarding them**. When activations age out of the fine-grained memory (FIFO buffer), they are mapped to a smaller set of "compressed memories" via a learned compression function, and stored in a secondary FIFO buffer. The model then attends over both the regular memory and the compressed memory using standard multi-head attention.

**Key equation:** The compression function `f_c : R^{n_s x d} -> R^{floor(n_s/c) x d}` maps `n_s` old memories down by a factor `c`, the compression rate.

**Temporal range:** With memory size `n_m` and compressed memory size `n_cm`, the max temporal range becomes `l * (n_m + c * n_cm)` — significantly longer than TransformerXL's `l * n_m` at similar attention cost `O(n_s^2 + n_s * (n_m + n_cm))`.

---

## Architecture Details

### Two-Tier Memory
1. **Memory (fine-grained):** FIFO of recent hidden activations (per layer), exactly like TransformerXL
2. **Compressed Memory (coarse):** FIFO of compressed older activations, created by applying `f_c` to evicted memories

At each step:
- The model processes a window of `n_s` tokens
- Old memory activations are evicted from the memory FIFO
- Instead of being discarded, they are compressed and pushed into the compressed memory FIFO
- Attention operates over `[compressed_memory | memory | current_sequence]`

### Compression Functions Tested
| Function | Loss | Enwik8 BPC |
|---|---|---|
| Conv + BPTT | Task loss only | 0.996 |
| Max Pooling | N/A | 0.986 |
| Conv + Auto-encoding | Reconstruction | 0.984 |
| Mean Pooling | N/A | 0.982 |
| Most-used (by attention weight) | N/A | 0.980 |
| Dilated conv + Attention-reconstruction | Lossy | 0.977 |
| **Conv + Attention-reconstruction** | **Lossy** | **0.973** |

**Winner:** 1D convolution with kernel/stride = compression rate `c`, trained with an **attention-reconstruction loss**. This is a *lossy* objective — it only preserves information that the model attends to, discarding unused content.

### Attention-Reconstruction Loss
A local auxiliary loss that trains the compression function to preserve attention patterns:
```
L_attn = || attn(h, old_mem) - attn(h, compressed_mem) ||_2
```
- Uses content-based attention (no relative positional embeddings)
- Gradients are **stopped** from flowing into the main transformer — only the compression network is optimized by this loss
- This decoupling is critical: the transformer optimizes the task objective, the compressor optimizes compression conditioned on task-relevant representations

---

## Key Results

### Language Modelling
| Benchmark | TransformerXL | Compressive Transformer |
|---|---|---|
| **Enwik8** (char-level BPC) | 0.98 | **0.97** (SotA at time) |
| **WikiText-103** (word-level PPL) | 18.1 | **17.1** (SotA at time) |
| **PG-19** (book-level PPL) | 36.3 | **33.6** |

### Rare Word Improvement
The biggest gains are on **rare/infrequent words**:
- Common words (>10K occurrences): 2.6% improvement
- Rare words (<100 occurrences): **~20% improvement**

This makes sense — rare words benefit most from long-range context that captures their earlier usage.

### Speech and RL
- **Speech:** Competitive with WaveNet on raw 24kHz audio (compression rate 4 worked best)
- **RL:** Solved a DMLab object matching memory task to human level as a drop-in LSTM replacement in IMPALA

---

## Important Findings

### 1. Compressed memory IS used
Attention analysis shows an **increase** in attention weight at the transition from regular memory to compressed memory. This goes against the expected trend of older memories being accessed less, providing evidence that the network learns to preserve salient information during compression.

### 2. Compression rate c=3-4 works best
Across all experiments, compression rates of 3 or 4 consistently outperformed lower rates.

### 3. Optimization quirk for long-context models
Reducing learning rate during training causes catastrophic performance drops for both TransformerXL and Compressive Transformer. This is due to distributional shift between training mode (parameters continuously updating) and eval mode (fixed parameters).

**Solution:** Reduce optimization *frequency* instead of learning rate. Updating parameters every 4 steps (effectively increasing batch size) near the end of training gives better generalization. This improved even the TransformerXL baseline from 0.995 to 0.984 BPC on Enwik8.

### 4. Layer compressibility
- The first layer is highly compressible
- No clear trend of compression difficulty increasing with depth
- Some non-contiguous layers have similar compression loss, suggesting information routing via skip connections

---

## PG-19 Benchmark
The paper also introduces **PG-19**, a book-level language modelling benchmark from Project Gutenberg (books published before 1919):
- **28,752 books**, 11GB of text
- Average document length: **69K words** (vs WikiText-103's 3.6K)
- Open vocabulary (no unk-ing)
- Over double the size of existing LM benchmarks

---

## Connections to nanochat and Triplet Compression

### Direct Relevance
This paper is referenced in the `doc/idea.md` triplet compression proposal. The Compressive Transformer represents one point on the spectrum of memory compression approaches, and understanding its design choices illuminates both what works and what the triplet approach could improve upon.

### What the Compressive Transformer Gets Right (and nanochat could adopt)
1. **Two-tier memory is sound:** The idea of keeping recent context at full fidelity and compressing older context is validated. This directly supports the triplet proposal's "two-zone" design (raw token window + triplet memory).
2. **Lossy compression > lossless:** The attention-reconstruction loss (preserve what's attended to, discard the rest) outperformed auto-encoding (preserve everything). This suggests triplet extraction should focus on **task-relevant facts**, not exhaustive extraction.
3. **Compression rate 3-4 is practical:** For triplets, this suggests each triplet should encode ~3-4 tokens' worth of information to be in the sweet spot.

### Where Triplet Compression Could Improve
1. **Interpretability:** Compressive Transformer compressed memories are opaque `d`-dimensional vectors. You cannot inspect them to see what information was preserved. Triplets are readable: `(Biden, signed, Bill)`.
2. **Entity boundaries:** Mean/max pooling or convolution over a passage mentioning both Alice and Bob produces a blurred vector. Triplets maintain distinct entities.
3. **Composability:** Two compressed memory buffers cannot be meaningfully merged. Two triplet sets can be merged via set union.
4. **Structured vs. unstructured:** The paper uses generic neural compression (convolutions). Triplets encode structured relations, which could provide stronger inductive bias for reasoning tasks.

### Implementation Lessons for nanochat

#### Attention Mask Design
The Compressive Transformer concatenates `[compressed_memory | memory | sequence]` and attends over all of it with standard multi-head attention. nanochat's triplet design would similarly concatenate `[triplet_embeddings | token_embeddings]`. The key architectural parallel:

```
# Compressive Transformer:  [cm | m | seq] -> attend over all
# Triplet proposal:         [triplets | tokens] -> tokens attend to triplets (read-only)
```

The triplet proposal adds a constraint: triplets are read-only (tokens attend to triplets, but not vice versa). The Compressive Transformer doesn't impose this — compressed memories can attend to everything. This is a design choice worth ablating.

#### Where It Maps to nanochat Code

The Compressive Transformer's memory management happens **between** forward passes (evict old activations, compress them, store). In nanochat:

- **`gpt.py:388-423` (forward pass):** The triplet approach modifies this once — prepend triplet embeddings to token embeddings. No inter-step memory management needed if using offline pre-extraction (Strategy A from `doc/idea.md`).
- **`flash_attention.py:99-100` (attention):** nanochat currently uses `flash_attn_func(q, k, v, causal=True, window_size=...)`. Adding triplets means the attention mask needs a block structure: causal within tokens, full attention from tokens to triplets, and either causal or full among triplets themselves.
- **`gpt.py:260-287` (window sizes):** The sliding window attention pattern (`SSSL`) would need to account for triplet positions. Triplets should likely always be in the attention window regardless of the sliding window size.

#### The Auxiliary Loss Question
The Compressive Transformer's best results use an attention-reconstruction auxiliary loss for the compression function. For triplet compression:
- **If using offline pre-extraction (Strategy A):** No auxiliary loss needed — triplets are pre-computed, the model just learns to use them via the standard next-token prediction loss.
- **If using self-extraction (Strategy C):** An auxiliary loss might help. Could train the model to emit triplets that preserve attention patterns from the full-context baseline, similar to the attention-reconstruction approach here.

#### The Optimization Insight
The finding about reducing optimization frequency instead of learning rate is directly applicable to nanochat training. nanochat's `base_train.py` already uses gradient accumulation, but the specific trick of **increasing accumulation steps later in training** (effectively growing batch size) could be worth trying regardless of the triplet work.

---

## Summary Table: Compressive Transformer vs. Triplet Compression

| Dimension | Compressive Transformer | Triplet Compression |
|---|---|---|
| **What is compressed** | Hidden activations (per-layer) | Text content → KG triplets |
| **Compression is** | Learned (convolution) | External (extraction model) or self-generated |
| **Compressed representation** | Opaque d-dim vectors | Interpretable (subject, relation, object) |
| **Compression rate** | Fixed c=3-4 | Variable, 5-200x depending on content |
| **Entity preservation** | No (pooling blurs entities) | Yes (distinct triplets per entity) |
| **Composability** | No | Yes (set union) |
| **Architecture change** | Add compressed memory FIFO + compression fn per layer | Add TripletEncoder, modify forward pass once |
| **Training change** | Auxiliary compression loss + BPTT | None (if offline extraction) |
| **Memory overhead** | ~2x (two FIFOs per layer) | ~10-15% (small triplet buffer) |
| **Proven at scale** | Yes (WikiText-103, Enwik8, PG-19, speech, RL) | Not yet (proposal stage) |

---

## Key Takeaway

The Compressive Transformer validates the core premise of the triplet compression proposal: **compressing older context works, and the model learns to use it**. The specific finding that attention-based lossy compression outperforms lossless approaches suggests that the triplet approach — which is inherently lossy but structured — is on the right track. The main gap is that the Compressive Transformer has been validated at scale, while triplet compression remains a proposal. The incremental validation plan in `doc/idea.md` (d12 first, then scale) is the right approach.

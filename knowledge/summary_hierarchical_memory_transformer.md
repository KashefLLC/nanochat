# HMT: Hierarchical Memory Transformer for Efficient Long Context Language Processing

**Paper:** He, Cao, Qin, Prakriya, Sun, Cong (UCLA / UCSD)
**Venue:** NAACL 2025
**arXiv:** [2405.06067](https://arxiv.org/abs/2405.06067)
**Code:** [github.com/OswaldHe/HMT-pytorch](https://github.com/OswaldHe/HMT-pytorch)

---

## Core Idea

HMT is a **model-independent plug-and-play framework** that augments any decoder-only LLM with hierarchical memory inspired by human cognition. It splits input into segments, processes them recurrently, and maintains a three-tier memory hierarchy:

1. **Sensory memory** — last `k` token embeddings from the previous segment (bridge between segments)
2. **Short-term memory** — a single embedding summarizing the current augmented segment
3. **Long-term memory** — a cache of `N` past segment summary embeddings, searchable via cross-attention

The key innovation over prior work (RMT, Compressive Transformer) is the **memory retrieval mechanism**: instead of just passing compressed state forward sequentially (which dilutes old information), HMT performs a cross-attention search over all cached memory embeddings to retrieve information relevant to the current segment.

---

## Architecture

### Four-Step Per-Segment Workflow

```
For each segment n:
  1. Representation Encoding:
     - Take first j tokens of segment + learnable summarization prompt T
     - Run through backbone model → summary embedding S_n

  2. Memory Search (cross-attention):
     - Q = S_n * W_q
     - K = M_[n-N+1,n) * W_k  (cached memory embeddings)
     - P_n = softmax(QK^T / sqrt(d)) * M_[n-N+1,n)
     → Memorization prompt embedding P_n (relevant past info)

  3. Prepend Sensory Memory:
     - Augmented input = [P_n | last_k_tokens_prev_segment | H_n | P_n]

  4. Decoding & Summarization:
     - Run backbone model on augmented input
     - Output: logits H_n^out + memory embedding M_n
     - Push M_n into long-term memory cache
```

### Key Design Choices

- **No architecture modification of the backbone model.** HMT wraps around any decoder-only model. The only new parameters are: summarization prompt `T`, cross-attention projections `W_q` and `W_k`. This is 0.5%-1.3% extra parameters.
- **Memory search uses simplified cross-attention**: no value/output projection — the softmax-weighted sum is applied directly to memory embeddings, keeping distributions aligned.
- **Partial summarization**: Only the first half of a segment (`j = L/2`) is used for representation encoding. This allows overlapping the encoding with inference on the previous segment, with negligible quality loss.
- **Sensory memory = 32 tokens** from the previous segment provides the best quality (too few loses continuity, too many wastes capacity).
- **Long-term memory size N = 300** is sufficient for 100K token inputs. Benefits of increasing N are diminishing.
- **Memory complexity O(l_i)** — constant regardless of input length, unlike LongMem (O(l_m)) or HOMER (O(log L)).

### Multi-Stage Training

HMT trains in two stages via BPTT:
1. **Stage 1** (200 steps): Train without memory retrieval, 2 segments unrolled — learn basic segment summarization
2. **Stage 2** (500 steps): Add memory retrieval, unroll maximum segments (up to 15) — learn to use long-term memory

This two-stage approach converges faster and produces better results than single-stage training.

---

## Key Results

### Language Modeling (Perplexity)

| Model | Wikitext-103 (30K tok) | PG-19 (60K tok) |
|---|---|---|
| OPT 2.7B baseline | 12.12 | — |
| HMT + OPT 2.7B | **8.61** (-28.9%) | — |
| RWKV 3B baseline | 13.13 | — |
| HMT + RWKV 3B | **9.93** (-25.3%) | — |
| HMT + Llama 2 7B | — | **7.40** |
| Mistral 7B (32K ctx) | 5.47 | — |
| HMT + Mistral 7B | **5.12** | — |
| Qwen 2.5 14B baseline | — | ~10.0 |
| HMT + Qwen 2.5 14B | — | ~9.0 (-10%) |

### vs. Prior Memory-Augmented Methods

| Comparison | HMT Advantage |
|---|---|
| vs. RMT | 13% better (Wikitext), 4-7% better (PG-19). RMT degrades RWKV; HMT improves it. RMT fails at scale (Qwen 14B); HMT still works. |
| vs. Memorizing Transformer | 13.67 vs 31.51 PPL on Wikitext (OPT 350M). HMT: O(l_i) memory; MemTRM: O(L). |
| vs. LongMem | 9.02 vs 10.08 PPL on ArXiv. HMT needs only 300 memory slots vs 65K. |
| vs. CCM-concat | Comparable PPL (7.40 vs 7.41 on PG-19), but HMT has O(l_i) memory vs O(t + l_i). |
| vs. HOMER | 9.9% better PPL on PG-19. HMT: O(1) peak memory; HOMER: O(log L). |

### vs. Large Long-Context Models (LongBench)

HMT + small models (OPT 350M, OpenLlama 3B, SmolLM 135M) achieve **comparable or superior** results to 6-7B long-context models on single/multi-document QA, with:
- **2-57x fewer parameters**
- **2.5-116x less inference memory**

Particularly strong on short-answer QA (context filtering ability). Weaker on long-response generation tasks (small backbone limitation).

### Gradient Stability

A critical advantage over RMT: **HMT does not suffer from gradient vanishing/explosion as BPTT depth increases.** RMT performance peaks at 5 segments then degrades. HMT continues improving up to 15 segments. The memory retrieval mechanism creates multiple short gradient branches instead of one long chain, preventing the classic RNN gradient problem.

---

## Memory Retrieval Behavior Analysis

The paper provides insightful analysis of what HMT's memory search actually does:

1. **Most retrieval is local** (6.5% within 2 segments), but long-range retrieval exists — Wikipedia entries referencing other entries show this.
2. **Context-switching detection**: When PG-19 samples are interleaved (alternating between two documents every 256 tokens), HMT's retrieval shows a periodic pattern matching the interleaving. RMT degrades on this task; HMT improves.
3. **Irrelevant content filtering**: When dollar signs are inserted to dilate samples, HMT learns to avoid retrieving those segments entirely.
4. **Correct context identification**: On PubMedQA with multiple contexts, HMT retrieves the correct original context for each question.

---

## Connections to nanochat and Triplet Compression

### Architectural Parallels

HMT and the triplet compression proposal share the same fundamental insight: **augment the current context window with compressed representations of older content**. The comparison:

| Dimension | HMT | Triplet Compression |
|---|---|---|
| **Compression unit** | Segment → single d_model embedding | Text → structured (subject, relation, object) |
| **Memory type** | Opaque learned embeddings | Interpretable KG triplets |
| **Retrieval** | Cross-attention over cached embeddings | All triplets prepended (attend to all) |
| **Selectivity** | Retrieves relevant past only | All triplets always visible |
| **Backbone requirement** | Wraps around existing model | Modifies forward pass |
| **Training** | Multi-stage BPTT, 0.5-1.3% extra params | Standard next-token prediction |
| **Entity preservation** | No (each embedding is a segment summary) | Yes (one triplet per fact) |
| **Composability** | No (embeddings are model-specific) | Yes (set union) |

### What nanochat Can Learn from HMT

#### 1. The Memory Retrieval Mechanism is Powerful
HMT's biggest win over RMT and Compressive Transformer is **selective retrieval** — not all past context is equally relevant. For triplet compression, this suggests:
- Don't just prepend all triplets. Consider a **retrieval step** that selects the most relevant triplets for the current window.
- This could be implemented as a lightweight cross-attention between a summary of the current tokens and the triplet embeddings, similar to HMT's memory search.
- In nanochat, this would be a small additional module before the main forward pass in `gpt.py`.

#### 2. Sensory Memory (Token Bridging) Matters
HMT finds that keeping 32 raw tokens from the previous segment significantly helps. This validates the triplet proposal's "two-zone" design — you need both compressed memory (triplets) AND recent raw tokens for good performance. The optimal sensory memory size (32 tokens, ~12% of segment) is an interesting data point.

#### 3. Partial Summarization Works
Using only the first half of a segment for representation encoding has negligible quality impact. For triplet extraction (Strategy C — self-extraction), this means the model might only need to "look at" the beginning of a text chunk to decide which triplets to extract.

#### 4. Multi-Stage Training is Effective
HMT's two-stage training (learn basic compression first, then learn retrieval) suggests a training curriculum for triplet compression:
- **Stage 1**: Train with triplets always prepended, no selection — learn to read triplet embeddings
- **Stage 2**: Add a retrieval/selection mechanism — learn which triplets are relevant

#### 5. Gradient Stability Through Short Paths
HMT's key advantage over RMT is that the cross-attention creates short gradient paths. For triplet compression, since triplets are pre-computed (offline extraction) and prepended as input, there is **no recurrence** and thus no gradient stability issue at all. This is a significant advantage of the triplet approach.

### Where Triplet Compression Could Improve on HMT

1. **Interpretability**: HMT's memory embeddings are opaque. When the model makes an error, you cannot inspect what the memory contains. Triplets are readable.

2. **Information density**: Each HMT memory embedding summarizes one segment (~256-1024 tokens) into one d_model vector. This is aggressive compression that necessarily loses information. Triplets preserve discrete facts with no information blurring.

3. **Cross-session memory**: HMT's embeddings are tied to the model's hidden state space — they cannot be transferred between sessions or models. Triplets are model-independent structured data that can persist.

4. **No BPTT needed**: HMT requires expensive multi-segment BPTT training (up to 15 segments). Triplet compression with offline extraction trains with standard next-token prediction — no special training procedure.

5. **Not model-wrapping**: HMT wraps around a backbone model, processing segments sequentially. This adds latency (multiple forward passes per long document). Triplets are prepended once and processed in a single forward pass.

### Possible Hybrid: HMT-style Retrieval + Triplets

The most interesting synthesis would combine both approaches:
- Use **triplets** as the memory representation (interpretable, structured)
- Use **HMT-style cross-attention retrieval** to select relevant triplets per segment
- This avoids prepending ALL triplets (which wastes attention on irrelevant facts) while keeping the interpretability and composability advantages

In nanochat code, this would look like:
```python
# In gpt.py forward():
# 1. Encode all available triplets
all_triplet_embeds = self.triplet_encoder(all_triplets)  # (B, N_triplets, d_model)

# 2. Summarize current tokens (first half, like HMT)
token_summary = self.summarizer(x[:, :T//2, :])  # (B, 1, d_model)

# 3. Cross-attention retrieval (select relevant triplets)
relevant_triplets = self.memory_search(token_summary, all_triplet_embeds)  # (B, 1, d_model)

# 4. Prepend to token embeddings
x = torch.cat([relevant_triplets, x], dim=1)
```

### Practical Implementation Insights

- **Segment length**: HMT uses 256-1024 tokens per segment. nanochat's default sequence_len is 2048. For triplet-augmented training, the raw token window should be at least 512-1024 tokens.
- **Memory budget**: 300 embeddings is sufficient for 100K tokens. For triplets, this means ~300 triplets should capture the relevant facts from ~100K tokens of prior context.
- **Training cost**: HMT trains in 700 steps with batch size 2. Very lightweight. Triplet-augmented nanochat would train with the standard budget (~3 hours 8xH100 for GPT-2), not requiring any special training procedure beyond the offline triplet preparation.
- **Scaling**: HMT shows consistent improvement from 135M (SmolLM) to 14B (Qwen 2.5). This suggests that triplet compression, if it works at d12, should continue working at d20-d26.

---

## Summary

HMT represents the current state-of-the-art in memory-augmented transformers. Its key contributions — hierarchical memory, cross-attention retrieval, gradient-stable BPTT — solve real problems in long-context processing. For the triplet compression project, HMT validates that:

1. **Compressed memory works** — models learn to use summarized past context
2. **Selective retrieval is critical** — not all past context is equally useful
3. **Small parameter overhead (0.5-1.3%) can enable big gains** — you don't need to reshape the whole model
4. **The approach scales** — from 135M to 14B backbone models

The triplet approach has potential advantages in interpretability, composability, and training simplicity (no BPTT). The ideal system might combine HMT's retrieval mechanism with triplet representations — getting both the selectivity of HMT and the structure of knowledge graph triplets.

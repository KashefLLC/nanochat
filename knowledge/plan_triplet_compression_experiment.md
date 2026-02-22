# Triplet Context Compression: Complete Research Plan

**Hardware:** NVIDIA RTX 3090 (24GB VRAM), 32GB RAM, 8-core CPU, CUDA 12.4
**Target Model:** nanochat d12 (768 dim, 6 heads, 12 layers, ~135M params)
**Estimated Total Time:** ~2-3 weeks of development + experiments
**Estimated Compute Cost:** $0 (all on local GPU)

---

## Research Synthesis: What We Learned from the Literature

### From Compressive Transformer (Rae et al., ICLR 2020):
1. **Lossy compression > lossless** — attention-reconstruction loss (preserve only what's attended to) beat auto-encoding (preserve everything). Triplet extraction should focus on task-relevant facts, not exhaustive extraction.
2. **Compression rate c=3-4 is optimal** — each compressed slot should represent ~3-4 tokens worth of information.
3. **Compressed memory IS used** — attention weight actually *increases* at the compressed memory boundary. Models learn to read compressed representations.
4. **Rare words benefit most** — 20% improvement on rare words vs 2.6% on common words. Triplets carrying entity names (inherently rare) should disproportionately help.

### From HMT (He et al., NAACL 2025):
1. **Selective retrieval is critical** — HMT's biggest win over flat approaches (RMT, Compressive Transformer) is that it retrieves only relevant past memory, not all of it. Blindly prepending all triplets may waste attention budget.
2. **0.5-1.3% extra parameters suffice** — massive architectural changes are unnecessary. A small TripletEncoder + optional retrieval module is enough.
3. **Multi-stage training helps** — learn basic compression first, then add retrieval. Apply this: train with all triplets first, then add selection.
4. **32 sensory tokens optimal** — HMT's best bridging between segments uses 32 raw token embeddings. Informs our raw token window sizing.
5. **No gradient stability issues without recurrence** — since our triplets are prepended (not recurrently passed), we avoid the gradient vanishing/explosion that plagues RMT.

### Improved Design (vs. original doc/idea.md proposal):

**Original:** Prepend ALL triplets blindly as read-only memory.
**Improved:** Two-phase approach:
- **Phase 1 (Baseline):** Prepend all triplets (validate the model uses them at all)
- **Phase 2 (Selective):** Add lightweight cross-attention retrieval (select top-K relevant triplets per batch), inspired by HMT

**Original:** Single MLP encoder for triplets.
**Improved:** Use a small 2-layer MLP with residual connection + learned positional encoding. Each triplet = 3 tokens worth of semantic info (subject, relation, object). Encode each component with shared entity/relation embeddings, concat, project to d_model.

**Original:** Temporal position encoding via additive embedding.
**Improved:** Use relative temporal encoding (how far back in tokens was this fact extracted), not absolute position. This generalizes better to variable-length contexts. Inspired by HMT's retrieval distance patterns.

---

## Phase 0: Establish Baseline (Days 1-2)

### Goal
Train a standard nanochat d12 model on your RTX 3090 as the reference point. All future comparisons use this baseline.

### Steps

#### 0.1 Environment Setup
```bash
cd /home/para2x/Documents/Github/nanochat
uv venv && source .venv/bin/activate
uv sync --extra gpu
```

#### 0.2 Data Download & Tokenizer Training
```bash
# Download 8 parquet shards (~2B characters, enough for d12)
python -m nanochat.dataset -n 8

# Train BPE tokenizer (vocab size 32768)
python -m scripts.tok_train
```

#### 0.3 Train Baseline d12
```bash
python -m scripts.base_train \
    --depth=12 \
    --device-batch-size=4 \
    --run="baseline_d12" \
    --eval-every=250 \
    --core-metric-every=1000 \
    --sample-every=1000
```

**Expected:**
- VRAM: ~6-8 GB (well within 24GB)
- Batch: 4 * 2048 = 8192 tokens/step, grad accum to reach ~524K effective batch
- Training time: ~2-4 hours for meaningful convergence
- Metrics: BPB (bits per byte), CORE score, sample quality

#### 0.4 Record Baseline Metrics
After training, record:
- [ ] Final validation BPB
- [ ] CORE metric score (22-task ensemble)
- [ ] Per-task CORE breakdown (especially knowledge-dependent tasks: ARC, MMLU, TriviaQA)
- [ ] Generated samples (qualitative coherence)
- [ ] Training time and tokens processed
- [ ] Peak VRAM usage

---

## Phase 1: Triplet Data Preparation (Days 3-5)

### Goal
Extract knowledge graph triplets from the FineWeb training data, creating a parallel triplet dataset.

### 1.1 Extraction Strategy: REBEL Model

Use the REBEL model (Relation Extraction By End-to-end Language generation) — a T5-based seq2seq model fine-tuned for open information extraction. It runs on CPU/GPU and outputs structured triplets directly.

```python
# scripts/triplet_extract.py
"""
Extract (subject, relation, object) triplets from FineWeb parquet shards.
Uses REBEL model for open information extraction.
Outputs parallel parquet files with triplet annotations.
"""
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

MODEL_NAME = "Babelscape/rebel-large"  # ~770M params, fits on RTX 3090

def extract_triplets_from_text(text, model, tokenizer, max_length=512):
    """Extract triplets from a text chunk using REBEL."""
    inputs = tokenizer(text, max_length=max_length, truncation=True, return_tensors="pt")
    outputs = model.generate(**inputs.to(model.device), max_length=256, num_beams=3)
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    return parse_rebel_output(decoded[0])

def parse_rebel_output(text):
    """Parse REBEL's structured output into (subject, relation, object) tuples."""
    triplets = []
    for triple_str in text.split("<triplet>")[1:]:
        parts = triple_str.split("<subj>")
        if len(parts) < 2: continue
        subj = parts[1].split("<obj>")[0].strip()
        rest = parts[1].split("<obj>")
        if len(rest) < 2: continue
        obj_rel = rest[1].strip()
        # REBEL format: <subj> subject <obj> object <rel> relation
        # Parse accordingly
        triplets.append({"subject": subj, "relation": rel, "object": obj})
    return triplets
```

### 1.2 Processing Pipeline

For each FineWeb parquet shard:
1. Read documents
2. Split into ~512-token chunks (REBEL's context window)
3. Extract triplets per chunk
4. Deduplicate triplets per document (UPSERT: newer overrides older for same subject+relation)
5. Store as parallel parquet: `triplets_shard_XXXXX.parquet`

**Schema:**
```
document_id: int
chunk_idx: int
triplets: list[{subject: str, relation: str, object: str}]
token_offset: int  (where in the document this chunk starts)
```

### 1.3 Build Entity/Relation Vocabularies

After extraction, build vocabularies:
```python
# Collect all unique entities and relations across the corpus
entity_vocab = sorted(set(all_subjects + all_objects))  # ~50K-200K entities
relation_vocab = sorted(set(all_relations))              # ~5K-20K relations

# Add special tokens: <PAD>, <UNK>, <NONE>
# Save as JSON for the model to load
```

**Key decision:** Cap entity vocab at ~50K and relation vocab at ~5K. Entities beyond the cap get mapped to <UNK> or hashed into buckets. This keeps embedding tables manageable.

### 1.4 Extraction Quality Check

Before training, validate extraction quality:
- Sample 100 documents, manually inspect triplets
- Compute: triplets per document (expect 5-50)
- Compute: unique entities per document
- Check for: coreference failures, negation handling, garbage triplets
- **Go/no-go decision:** If >50% of triplets are garbage, try a different extractor (e.g., prompted Qwen-2.5 7B) before proceeding

---

## Phase 2: Model Architecture Changes (Days 6-8)

### Goal
Add TripletEncoder to nanochat's GPT model and modify the forward pass.

### 2.1 TripletEncoder Module

```python
# Add to nanochat/gpt.py

class TripletEncoder(nn.Module):
    """Encode (subject, relation, object) triplets into d_model vectors."""

    def __init__(self, n_entities, n_relations, n_embd, max_triplets=128):
        super().__init__()
        entity_dim = n_embd // 3  # Split d_model across 3 components
        rel_dim = n_embd - 2 * entity_dim  # Remainder goes to relation

        self.entity_embed = nn.Embedding(n_entities, entity_dim)
        self.relation_embed = nn.Embedding(n_relations, rel_dim)

        # Project concatenated (subj, rel, obj) to d_model
        self.proj = nn.Linear(n_embd, n_embd, bias=False)

        # Relative temporal position encoding (how far back was this fact)
        self.temporal_embed = nn.Embedding(max_triplets, n_embd)

    def forward(self, triplet_ids, temporal_positions):
        """
        triplet_ids: (B, N_triplets, 3) — [subject_id, relation_id, object_id]
        temporal_positions: (B, N_triplets) — relative position (0=most recent)
        Returns: (B, N_triplets, d_model)
        """
        subj = self.entity_embed(triplet_ids[:, :, 0])
        rel = self.relation_embed(triplet_ids[:, :, 1])
        obj = self.entity_embed(triplet_ids[:, :, 2])

        x = torch.cat([subj, rel, obj], dim=-1)  # (B, N, d_model)
        x = self.proj(x)
        x = x + self.temporal_embed(temporal_positions)
        return norm(x)
```

### 2.2 Modified GPT Forward Pass

```python
# In GPT.forward(), modify to accept optional triplets:

def forward(self, idx, targets=None, triplet_ids=None, triplet_positions=None,
            kv_cache=None, loss_reduction='mean'):
    B, T = idx.size()

    # Standard token embeddings
    x = self.transformer.wte(idx)
    x = norm(x)

    # Prepend triplet embeddings if provided
    N_triplets = 0
    if triplet_ids is not None:
        triplet_embeds = self.triplet_encoder(triplet_ids, triplet_positions)
        N_triplets = triplet_embeds.size(1)
        x = torch.cat([triplet_embeds, x], dim=1)  # (B, N+T, d_model)

    x0 = x  # for x0 residual

    # Rotary embeddings: offset by N_triplets for token positions
    # Triplets get positions 0..N-1, tokens get N..N+T-1
    T_total = N_triplets + T
    cos_sin = self.cos[:, :T_total], self.sin[:, :T_total]

    for i, block in enumerate(self.transformer.h):
        x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        # Value embeddings only for token positions (not triplets)
        ve = None
        if str(i) in self.value_embeds:
            # Pad triplet positions with zeros for value embed lookup
            padded_idx = F.pad(idx, (N_triplets, 0), value=0)
            ve_full = self.value_embeds[str(i)](padded_idx)
            ve = ve_full  # Block handles the slicing
        x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)

    x = norm(x)

    # Only compute logits on TOKEN positions (not triplet positions)
    x_tokens = x[:, N_triplets:, :]
    logits = self.lm_head(x_tokens)
    logits = logits[..., :self.config.vocab_size]
    logits = logits.float()
    logits = 15 * torch.tanh(logits / 15)  # softcap

    if targets is not None:
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1),
            ignore_index=-1, reduction=loss_reduction
        )
        return loss
    return logits
```

### 2.3 Attention Mask: Triplets as Read-Only

Modify `CausalSelfAttention.forward()` to support the block-causal mask:
- Tokens attend to: all triplets (bidirectional) + previous tokens (causal)
- Triplets attend to: other triplets only (bidirectional among themselves)
- Triplets do NOT attend to tokens

This is implemented by constructing an explicit attention mask when triplets are present:

```python
# In flash_attention.py or CausalSelfAttention:
def build_triplet_mask(N_triplets, T_tokens, device):
    """Block-causal mask for [triplets | tokens] layout."""
    total = N_triplets + T_tokens
    mask = torch.zeros(total, total, dtype=torch.bool, device=device)

    # Triplets attend to all triplets (bidirectional)
    mask[:N_triplets, :N_triplets] = True

    # Tokens attend to all triplets
    mask[N_triplets:, :N_triplets] = True

    # Tokens attend to previous tokens (causal)
    token_causal = torch.tril(torch.ones(T_tokens, T_tokens, dtype=torch.bool, device=device))
    mask[N_triplets:, N_triplets:] = token_causal

    return mask
```

**Note:** This custom mask is incompatible with FA3's built-in `causal=True`. We'll need to use SDPA with explicit mask on the RTX 3090 anyway (no FA3 on Ampere).

### 2.4 Parameter Budget

New parameters added:
- `entity_embed`: 50K * 256 = 12.8M params
- `relation_embed`: 5K * 256 = 1.3M params
- `proj`: 768 * 768 = 0.6M params
- `temporal_embed`: 128 * 768 = 0.1M params
- **Total: ~14.8M params (10.9% of base model)**

This is larger than HMT's 0.5-1.3%, but entity/relation embeddings are sparse (most are rarely accessed). Can reduce by capping entity vocab at 20K (5.2M) or using hash embeddings.

---

## Phase 3: Dataloader Modifications (Days 8-9)

### Goal
Modify the dataloader to yield `(triplets, tokens, targets)` per batch.

### 3.1 Triplet-Aware Dataloader

```python
def triplet_data_loader(base_loader, triplet_shards, max_triplets=64):
    """
    Wraps the base nanochat dataloader to add triplet context.

    For each batch of tokens, looks up the document's pre-extracted triplets
    and packages them alongside the token batch.
    """
    for (inputs, targets, state_dict) in base_loader:
        # Look up triplets for current documents
        triplet_ids = []  # (B, max_triplets, 3)
        triplet_positions = []  # (B, max_triplets)

        for doc_idx in batch_doc_indices:
            doc_triplets = lookup_triplets(triplet_shards, doc_idx)
            # Pad/truncate to max_triplets
            padded = pad_triplets(doc_triplets, max_triplets)
            triplet_ids.append(padded)
            positions = compute_temporal_positions(doc_triplets, max_triplets)
            triplet_positions.append(positions)

        triplet_ids = torch.stack(triplet_ids)
        triplet_positions = torch.stack(triplet_positions)

        yield inputs, targets, triplet_ids, triplet_positions, state_dict
```

### 3.2 Document-Triplet Alignment

The tricky part: nanochat's best-fit packing means each batch row may contain parts of multiple documents. We need to:

1. Track which document(s) each row comes from
2. For each row, gather triplets from the **preceding** document content (not the current window — that defeats the purpose)
3. Triplets represent older context that was "compressed away"

**Simplification for v1:** During data preparation, for each 2048-token training sample, extract triplets from the PREVIOUS 2048-4096 tokens of the same document. This simulates the "older context compressed into triplets" scenario without complex online tracking.

---

## Phase 4: Training (Days 9-12)

### Goal
Train the triplet-augmented d12 model and compare to baseline.

### 4.1 Training Configuration

```bash
python -m scripts.base_train \
    --depth=12 \
    --device-batch-size=4 \
    --use-triplets \
    --max-triplets=64 \
    --run="triplet_d12_v1" \
    --eval-every=250 \
    --core-metric-every=1000 \
    --sample-every=1000
```

**Expected VRAM impact:**
- Base d12: ~6-8 GB
- + Triplet embeddings (14.8M params): ~30 MB
- + Attention over 64+2048 positions instead of 2048: ~3% more
- + Custom attention mask: negligible
- **Total: ~7-9 GB** — well within 24GB

### 4.2 Training Schedule

1. **Quick sanity check** (10 min): Train 500 steps, verify loss decreases, no NaN
2. **Full training** (2-4 hours): Match baseline token budget (~700M tokens for d12)
3. **Ablation runs** (1-2 hours each):
   - Triplet model with triplets MASKED (zeroed) at eval — does PPL increase?
   - Triplet model with random/shuffled triplets — does the model learn to ignore noise?
   - Vary max_triplets: 16, 32, 64, 128

---

## Phase 5: Comprehensive Evaluation (Days 12-15)

### Goal
Compare baseline vs triplet model across multiple dimensions.

### 5.1 Standard Metrics (Automated)

| Metric | How | What It Tells Us |
|---|---|---|
| **Validation BPB** | Built-in eval | Overall language modeling quality |
| **CORE Score** | Built-in eval (22 tasks) | Downstream task performance |
| **Per-task CORE** | Built-in, breakdown | Which task categories benefit |
| **Training Loss Curve** | wandb | Convergence speed comparison |
| **Tokens/sec** | Built-in | Throughput impact of triplets |

### 5.2 Triplet-Specific Metrics (Custom)

#### 5.2.1 Triplet Utilization — "Does the Model Use Triplets?"

```python
def measure_triplet_utilization(model, eval_data):
    """
    Compare PPL with and without triplets on the same data.
    If PPL_without >> PPL_with, the model relies on triplets.
    """
    ppl_with_triplets = evaluate(model, eval_data, use_triplets=True)
    ppl_without_triplets = evaluate(model, eval_data, use_triplets=False)

    utilization = (ppl_without_triplets - ppl_with_triplets) / ppl_without_triplets
    return utilization  # >0 means triplets help, 0 means ignored
```

**Go/no-go:** If utilization < 1%, the model is ignoring triplets. Debug before proceeding.

#### 5.2.2 Attention Weight Analysis

```python
def analyze_triplet_attention(model, eval_data):
    """
    Extract attention weights from all layers/heads.
    Measure: what fraction of attention goes to triplet positions vs token positions?
    Inspired by Compressive Transformer's attention analysis.
    """
    # For each layer, each head:
    # avg_attention_to_triplets = mean(attn[:, :, N_triplets:, :N_triplets])
    # Plot by layer depth — do deeper layers attend more to triplets?
```

#### 5.2.3 Effective Context Length Test

Create a controlled experiment:
1. Take documents longer than 4096 tokens
2. **Baseline:** Feed last 2048 tokens (standard context)
3. **Triplet model:** Feed triplets from tokens 0-2048, then last 2048 raw tokens
4. Ask factual questions about content in tokens 0-2048
5. Compare answer accuracy

This directly tests whether triplets carry information from beyond the raw context window.

### 5.3 Hallucination Evaluation

#### 5.3.1 Grounded Generation Test

```python
def hallucination_test(model, test_cases):
    """
    Give the model a set of triplets + a question.
    Check if the answer is grounded in the triplets.

    Measures: does providing explicit structured facts reduce hallucination?
    """
    for triplets, question, ground_truth in test_cases:
        # Generate answer with triplets
        answer = model.generate(triplets + question)
        # Check: does the answer contain only facts from triplets?
        grounded = check_grounding(answer, triplets)
        # Compare: generate without triplets
        baseline_answer = model.generate(question)
        baseline_grounded = check_grounding(baseline_answer, triplets)
```

#### 5.3.2 Factual Consistency Score

For each generated passage:
1. Extract triplets from the generation using the same REBEL model
2. Compare extracted triplets to the input triplets
3. **Contradiction rate:** generated triplets that conflict with input triplets
4. **Fabrication rate:** generated triplets with entities not in input triplets

### 5.4 Token Efficiency Evaluation

#### 5.4.1 Information Density

```python
def information_density_test(baseline_model, triplet_model, eval_data):
    """
    For the same effective information, how many tokens does each model need?

    Test: Given a long document, what's the minimum context needed to answer
    factual questions about it?

    Baseline: needs raw tokens in context
    Triplet model: triplets carry the facts in fewer positions
    """
    for doc, questions in eval_data:
        triplets = extract_triplets(doc)

        # Baseline: binary search for minimum context length
        min_ctx_baseline = binary_search_min_context(baseline_model, doc, questions)

        # Triplet: triplets + minimum raw tokens
        min_ctx_triplet = binary_search_min_context(
            triplet_model, doc, questions, triplets=triplets
        )

        # Effective positions: triplet model uses N_triplets + min_ctx_triplet
        # vs baseline uses min_ctx_baseline
        compression_ratio = min_ctx_baseline / (len(triplets) + min_ctx_triplet)
```

#### 5.4.2 Perplexity per Attention Position

```
Metric: PPL / total_attention_positions
Baseline: PPL_base / 2048
Triplet: PPL_triplet / (N_triplets + 2048)

If PPL_triplet is close to PPL_base with fewer effective tokens,
the triplet model is more efficient per position.
```

### 5.5 Downstream Task Evaluation

Focus on tasks where triplet context should help most:

| Task | Why Triplets Should Help | Metric |
|---|---|---|
| **ARC** (science QA) | Scientific facts as triplets | Accuracy |
| **MMLU** (multitask) | Domain knowledge as triplets | Accuracy |
| **TriviaQA** | Factual recall from triplets | Exact match |
| **WinoGrande** | Entity disambiguation from triplets | Accuracy |
| **GSM8K** (math) | Should NOT help much (tests reasoning) | Accuracy (control) |
| **HumanEval** (code) | Should NOT help much (tests generation) | pass@1 (control) |

The math and code tasks are **controls** — if triplets improve these, something unexpected is happening (possibly just more parameters helping, not the triplets themselves).

### 5.6 Composability Test

```python
def composability_test(model):
    """
    Test: Can we merge triplets from two separate contexts and use them?

    1. Extract triplets from Document A
    2. Extract triplets from Document B
    3. Merge (set union)
    4. Feed merged triplets + a question that requires facts from BOTH documents
    5. Check if model can answer correctly
    """
    triplets_a = extract(doc_a)  # e.g., "Alice is an engineer at Google"
    triplets_b = extract(doc_b)  # e.g., "Google's HQ is in Mountain View"
    merged = triplets_a + triplets_b

    # Question: "Where does Alice work?" → "Mountain View" (requires both)
    answer = model.generate(merged + question)
```

---

## Phase 6: Ablation Studies (Days 15-18)

### 6.1 Triplet Count Ablation

| Config | N_triplets | Expected |
|---|---|---|
| No triplets | 0 | Baseline |
| Few triplets | 16 | Minimal context |
| Medium triplets | 64 | Default |
| Many triplets | 128 | Diminishing returns? |
| Max triplets | 256 | VRAM limit test |

### 6.2 Triplet Quality Ablation

| Config | Description | Expected |
|---|---|---|
| Gold triplets | Manually verified, high quality | Upper bound |
| REBEL triplets | Standard extraction | Normal |
| Noisy triplets | 30% of triplets are random | Degraded |
| Shuffled triplets | Correct triplets, wrong documents | Model should ignore |
| Empty triplets | All-zero triplet embeddings | ~Baseline |

### 6.3 Architecture Ablation

| Config | Description | Expected |
|---|---|---|
| Prepend (read-only) | Triplets don't attend to tokens | Default |
| Prepend (bidirectional) | Triplets attend to everything | Possibly better? |
| Interleave | Triplets mixed among tokens | Test alternative |
| Cross-attention only | HMT-style: generate query from tokens, retrieve from triplets | More selective |

### 6.4 Triplet Encoding Ablation

| Config | Description |
|---|---|
| Concat + Linear | Default: concat(subj, rel, obj) → Linear |
| Sum | Average embeddings of subj, rel, obj |
| Bilinear | Bilinear interaction between components |
| Text encoding | Encode triplet as text tokens ("Biden signed Bill") |

---

## Phase 7: Analysis and Write-Up (Days 18-21)

### 7.1 Expected Outcomes (Predictions)

**Optimistic scenario (triplets work well):**
- 5-15% BPB improvement on long-context evaluation
- 2-5% CORE improvement, concentrated in knowledge tasks (ARC, MMLU, TriviaQA)
- Clear triplet utilization (>10% PPL increase when triplets removed)
- Reduced hallucination rate on grounded generation tasks
- 2-3x effective context compression ratio

**Neutral scenario (triplets help marginally):**
- 1-3% BPB improvement
- Marginal CORE improvement (~1%)
- Triplet utilization is positive but small (~3-5%)
- Mixed hallucination results
- Conclusion: idea has merit but extraction quality is the bottleneck

**Negative scenario (triplets don't help):**
- No significant BPB improvement
- Triplet utilization near 0% (model ignores them)
- Shuffled triplets perform the same as correct triplets
- Conclusion: either (a) extraction quality too poor, (b) model too small to learn triplet reading, or (c) the idea fundamentally doesn't work at this scale. Diagnose which.

### 7.2 Consequence Prediction for Each Step

| Step | If Succeeds | If Fails | Mitigation |
|---|---|---|---|
| Baseline d12 | Have reference point | — | This should always work |
| Triplet extraction | High-quality triplets | Garbage triplets | Try prompted LLM extraction instead |
| TripletEncoder | Model trains normally | NaN/divergence | Reduce LR for new params, check initialization |
| Training convergence | Loss decreases, matches baseline | Loss plateaus above baseline | Debug: are triplets providing signal? Check attention weights |
| Triplet utilization | >5% PPL delta | <1% PPL delta | Force utilization: add auxiliary loss (predict triplet content from tokens) |
| CORE improvement | Knowledge tasks improve | No improvement | Triplets may not help at d12 scale. Need larger model. |
| Hallucination reduction | Grounded answers improve | No change | Triplets aren't specific enough. Improve extraction granularity |

### 7.3 Files to Produce

1. **Quantitative comparison table:** Baseline vs Triplet across all metrics
2. **Attention heatmaps:** Where does the model attend in triplet positions?
3. **Ablation summary:** Which factors matter most?
4. **Failure mode analysis:** When do triplets hurt or get ignored?
5. **Updated doc/idea.md:** Revise claims based on actual results

---

## Summary: Complete File Change List

| File | Change | Effort |
|---|---|---|
| `nanochat/gpt.py` | Add `TripletEncoder`, modify `forward()`, modify `GPTConfig` | ~150 lines |
| `nanochat/dataloader.py` | Add triplet-aware data loading | ~100 lines |
| `nanochat/flash_attention.py` | Add `build_triplet_mask()` for custom attention | ~30 lines |
| `scripts/base_train.py` | Add `--use-triplets`, `--max-triplets` args, modify training loop | ~50 lines |
| `scripts/triplet_extract.py` | **NEW:** Offline triplet extraction pipeline | ~200 lines |
| `scripts/triplet_eval.py` | **NEW:** Triplet-specific evaluation suite | ~300 lines |
| `nanochat/core_eval.py` | No changes (standard eval still works) | 0 |
| `nanochat/optim.py` | No changes (optimizer handles new params automatically) | 0 |

**Total: ~830 lines of new/modified code**

---

## Timeline Summary

| Days | Phase | Deliverable |
|---|---|---|
| 1-2 | Baseline | Trained d12 model + metrics |
| 3-5 | Data Prep | Triplet-annotated FineWeb dataset |
| 6-8 | Architecture | Modified gpt.py + dataloader |
| 9-12 | Training | Trained triplet d12 + sanity checks |
| 12-15 | Evaluation | Full comparison across all metrics |
| 15-18 | Ablations | Understanding what works and why |
| 18-21 | Analysis | Write-up, conclusions, next steps |

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| REBEL extraction quality too low | Medium | High | Fall back to prompted Qwen-2.5 or GPT-4 extraction |
| RTX 3090 OOM with 128 triplets | Low | Medium | Reduce to 64 or 32 triplets; reduce batch_size to 2 |
| Custom attention mask kills throughput | Medium | Medium | Profile; if >2x slower, use simpler prepend-only approach |
| Entity vocab too large for GPU memory | Low | Medium | Use hash embeddings or cap at 20K |
| Model ignores triplets entirely | Medium | High | Add auxiliary loss: predict masked triplet components from context |
| Training instability with new params | Low | Medium | Warm up TripletEncoder LR separately over 200 steps |

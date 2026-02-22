# Starting Point: nanochat vs SmolLM Fine-tuning


> **TL;DR:** For triplet context compression research, **start with nanochat**. You need control over the forward pass, data loading, and attention mechanism — all of which are locked behind HuggingFace abstractions in SmolLM. nanochat is 7.4K lines of hackable PyTorch; SmolLM is a 1.7B pre-trained checkpoint you'd integrate with transformers library boilerplate.


---



# Triplet Context Compression: Structured Memory for Transformers


> **Status:** Proposal / Early Research
> **Applies to:** nanochat GPT architecture (`nanochat/gpt.py`)
> **Core claim:** Replace raw token history with knowledge graph triplets as a structured, compressed context window — giving the model 3-40x more effective context at the same compute cost.


---


## 1. The Problem with Raw Token Context


Every transformer today burns its entire attention budget on raw tokens. A 2048-token context window in nanochat stores *everything* — function words, punctuation, repeated phrases, filler — with equal weight.


```mermaid
graph LR
   subgraph "Current: 2048 raw tokens"
       T1["The"] --> T2["president"]
       T2 --> T3["of"]
       T3 --> T4["the"]
       T4 --> T5["United"]
       T5 --> T6["States"]
       T6 --> T7[","]
       T7 --> T8["Joe"]
       T8 --> T9["Biden"]
       T9 --> T10[","]
       T10 --> T11["signed"]
       T11 --> T12["the"]
       T12 --> T13["infrastructure"]
       T13 --> T14["bill"]
       T14 --> T15["on"]
       T15 --> T16["Tuesday"]
       T16 --> T17["."]
   end


   style T1 fill:#ccc,stroke:#999
   style T3 fill:#ccc,stroke:#999
   style T4 fill:#ccc,stroke:#999
   style T7 fill:#ccc,stroke:#999
   style T10 fill:#ccc,stroke:#999
   style T12 fill:#ccc,stroke:#999
   style T15 fill:#ccc,stroke:#999
   style T17 fill:#ccc,stroke:#999
```


8 out of 17 tokens (the grey ones) carry near-zero semantic content. They're syntactic scaffolding. The model spends attention on "the", ",", and "." when what actually matters is: **Biden signed the infrastructure bill on Tuesday**.


Now multiply this waste across 2048 positions. At best, half the context window holds real information. The rest is structural noise.


---


## 2. The Idea: Triplets as Compressed Memory


What if older context was stored not as raw tokens, but as knowledge graph triplets?


```mermaid
graph LR
   subgraph "Proposed: Hybrid Context"
       direction LR
       subgraph TRIPLETS["Compressed Memory (triplets)"]
           direction TB
           TR1["(Biden, signed, InfrastructureBill)"]
           TR2["(Biden, title, President)"]
           TR3["(Signing, date, Tuesday)"]
       end
       subgraph RAW["Local Window (raw tokens)"]
           direction TB
           R1["He said the bill"]
           R2["would create millions"]
           R3["of new jobs in..."]
       end
   end
   TRIPLETS --> RAW
```


The model keeps a **short raw token window** (last 512-1024 tokens) for syntactic coherence and local context. Everything older gets **extracted into triplets** — a structured, compressed representation that preserves facts while discarding filler.


3 triplets replace 17 tokens. That's a **5.7x compression** on a single sentence. On longer passages, the ratio climbs to **20-40x**.


---


## 3. Architecture


### 3.1 The Two-Zone Attention Window


```mermaid
graph TB
   subgraph INPUT["Model Input"]
       direction LR
       subgraph ZONE_A["Zone A: Triplet Memory"]
           direction TB
           A1["Embedded triplets from older context"]
           A2["Each triplet = 1 vector in d_model space"]
           A3["Temporal position encoded"]
       end
       subgraph ZONE_B["Zone B: Raw Tokens"]
           direction TB
           B1["Recent 512-1024 tokens"]
           B2["Standard token embeddings"]
           B3["RoPE positional encoding"]
       end
   end


   subgraph TRANSFORMER["Transformer Blocks"]
       ATT["Causal Self-Attention"]
       MLP_BLOCK["MLP"]
   end


   subgraph OUTPUT["Output"]
       NEXT["Next token prediction"]
   end


   ZONE_A --> TRANSFORMER
   ZONE_B --> TRANSFORMER
   TRANSFORMER --> OUTPUT
```


**Attention rules:**
- Raw tokens attend to **both** triplets and other raw tokens (causal)
- Triplets are **read-only** — they don't attend to raw tokens (they're finalized summaries)
- Triplets attend to each other (entity relationships across the memory)


### 3.2 How Triplets Enter the Model


```mermaid
flowchart LR
   subgraph TRIPLET["Triplet (Biden, signed, Bill)"]
       S["Subject: Biden"]
       P["Predicate: signed"]
       O["Object: Bill"]
   end


   subgraph ENCODE["TripletEncoder (MLP)"]
       E_S["Entity Embed"]
       E_R["Relation Embed"]
       E_O["Entity Embed"]
       CONCAT["Concat"]
       PROJ["Linear → d_model"]
       TEMP["+ Temporal Position"]
   end


   subgraph OUT["Output"]
       VEC["Single vector in d_model space"]
   end


   S --> E_S --> CONCAT
   P --> E_R --> CONCAT
   O --> E_O --> CONCAT
   CONCAT --> PROJ --> TEMP --> VEC
```


Each triplet becomes **one position** in the attention window. A vocabulary of entities and relations is learned alongside the token vocabulary. The temporal position tells the model *when* in the conversation this fact was established — critical for tracking state changes ("Biden signed" at time T, "Biden vetoed" at time T+500).


### 3.3 Integration with nanochat's GPT


The change to `gpt.py` forward pass is minimal:


```mermaid
flowchart TD
   subgraph CURRENT["Current Forward Pass"]
       C1["idx → wte(idx) → token embeddings"]
       C2["+ RoPE positional encoding"]
       C3["→ Transformer blocks"]
       C4["→ lm_head → logits"]
   end


   subgraph PROPOSED["Proposed Forward Pass"]
       P1["idx → wte(idx) → token embeddings"]
       P2["+ RoPE positional encoding"]
       P3["triplets → TripletEncoder → triplet embeddings"]
       P4["+ Temporal position encoding"]
       P5["cat(triplet_embeds, token_embeds)"]
       P6["→ Transformer blocks (with modified attention mask)"]
       P7["→ lm_head → logits (on token positions only)"]
   end
```


The loss function, optimizer, and training loop stay **unchanged**. The model still does next-token prediction. It just has a richer, more structured context to predict from.


---


## 4. The Compression Advantage


### 4.1 Concrete Examples


**Example 1: Legal Document**


A 2048-token legal passage about a contract dispute:


```
Raw tokens: 2048 positions consumed
Key facts: ~12 (parties, dates, clauses, amounts, rulings)
```


As triplets:
```
(AliceCorp, sued, BobLLC)
(Lawsuit, filed_on, 2024-03-15)
(Contract, value, $2.4M)
(AliceCorp, claims, BreachOfContract)
(Clause7, requires, DeliveryBy2024-01)
(BobLLC, delivered_on, 2024-03-01)
(BobLLC, defense, ForceMajeure)
(Judge, ruled, InFavorOfAlice)
(Damages, amount, $800K)
(BobLLC, must_pay_by, 2024-12-31)
(AliceCorp, represented_by, SmithLaw)
(BobLLC, represented_by, JonesLLP)
```


**12 triplets = 12 positions. Compression: 170x.**


The model now has the remaining ~2036 positions for fresh raw tokens. Effective context: the information content of **~4000 raw tokens** in the space of 2048.


**Example 2: Multi-turn Conversation**


```
User: I'm working on a Python web app using FastAPI. The database is PostgreSQL.
     I need help with the authentication system. We're using JWT tokens.
     The frontend is React. Deployment is on AWS ECS.


[... 6 more turns discussing implementation details ...]
```


After 8 turns (~1500 tokens consumed), the triplet memory contains:


```
(Project, language, Python)
(Project, framework, FastAPI)
(Project, database, PostgreSQL)
(Project, auth_method, JWT)
(Project, frontend, React)
(Project, deployment, AWS_ECS)
(Auth, status, InProgress)
(User, needs_help_with, TokenRefresh)
```


**8 triplets instead of ~1500 tokens. Compression: 187x.**


Every future turn can reference "the project uses FastAPI" without re-reading 1500 tokens of conversation history. The model knows the full project context in 8 attention positions.


**Example 3: Story/Narrative (worst case)**


```
"The old man sat by the window, watching the rain trace paths
down the glass like tears on a weathered face."
```


As triplets:
```
(OldMan, action, Sitting)
(OldMan, location, ByWindow)
(Rain, action, Falling)
```


**Compression: ~5x — but the soul of the sentence is lost.** The simile, the mood, the imagery — gone. This is where triplet compression is weakest. The model would need the raw tokens for any literary, stylistic, or emotional content.


This is why the **two-zone design is essential** — raw tokens for what's recent and nuanced, triplets for what's older and factual.


---


### 4.2 Effective Context Scaling


```mermaid
graph LR
   subgraph CURRENT["Standard GPT (nanochat today)"]
       direction TB
       CUR_WIN["2048 raw tokens"]
       CUR_INFO["~1000 tokens of real information<br/>(rest is syntactic filler)"]
   end


   subgraph PROPOSED["With Triplet Compression"]
       direction TB
       PROP_TRIP["~100 triplets = 100 positions<br/>(representing ~4000-8000 tokens of facts)"]
       PROP_RAW["~1500 raw tokens<br/>(recent context, full fidelity)"]
       PROP_TOTAL["Effective information: ~5000-9000 tokens<br/>in 1600 attention positions"]
   end


   CURRENT -->|"3-5x more<br/>effective context"| PROPOSED
```


The attention cost stays roughly the same (similar number of positions), but the **information bandwidth** of the context window multiplies.


---


## 5. How It Compares


### 5.1 vs. Longer Context Windows


The brute-force alternative: just make the context longer (4096, 8192, 128K tokens).


```mermaid
graph TB
   subgraph APPROACH_A["Approach: Longer Context"]
       A1["4x context = 16x attention cost (quadratic)"]
       A2["All tokens treated equally"]
       A3["Model must learn to ignore filler"]
       A4["Needle-in-haystack: exact token is there"]
       A5["No information loss"]
   end


   subgraph APPROACH_B["Approach: Triplet Compression"]
       B1["Same attention cost as 2048 context"]
       B2["Filler eliminated at compression stage"]
       B3["Model attends to pre-extracted facts"]
       B4["Needle-in-haystack: depends on extractor quality"]
       B5["Lossy — style and nuance discarded"]
   end
```


Longer context is better when you need verbatim recall ("what was the exact wording?"). Triplet compression is better when you need **factual reasoning over large spans** ("what did the user say about authentication three pages ago?").


The key insight: **most real tasks need facts, not verbatim recall.** Coding assistance, Q&A, analysis, conversation — these are all fact-retrieval problems where triplet compression should win.


### 5.2 vs. Existing Memory Compression Approaches


```mermaid
graph TD
   subgraph METHODS["Memory Compression Methods"]
       CT["Compressive Transformer<br/>(Rae et al. 2019)<br/>Pools old tokens into<br/>averaged vectors"]
       HMT["HMT (NAACL 2025)<br/>Hierarchical segment<br/>summarization"]
       MEMOS["MemOS (2025)<br/>OS-level graph-structured<br/>memory management"]
       OURS["Triplet Compression<br/>(This proposal)<br/>KG triplets as structured<br/>compressed context"]
   end


   subgraph PROPERTIES["Key Differences"]
       direction TB
       P1["Interpretability"]
       P2["Entity distinction"]
       P3["Composability"]
       P4["Extraction difficulty"]
   end


   CT -->|"Opaque vectors.<br/>Averaging destroys<br/>entity boundaries."| P1
   CT -->|"'Apple' + 'Orange'<br/>= fruit mush vector"| P2


   HMT -->|"Learned embeddings.<br/>Slightly interpretable."| P1
   HMT -->|"Better than pooling,<br/>still not explicit."| P2


   OURS -->|"Fully readable.<br/>(Biden, signed, Bill)"| P1
   OURS -->|"Each entity is a<br/>distinct node."| P2
   OURS -->|"Two graphs can be<br/>merged via set union."| P3
   OURS -->|"Requires relation<br/>extraction model."| P4
```


**The core advantage over pooling/averaging:** triplets maintain **entity boundaries**. If the context mentions both Alice and Bob with different attributes, a pooled vector blurs them together. Triplets keep `(Alice, role, Engineer)` and `(Bob, role, Designer)` as distinct, addressable facts.


**The core advantage over learned memory embeddings:** triplets are **interpretable and debuggable**. When the model gets something wrong, you can inspect the triplet memory and see exactly what information it had available. With opaque embeddings, you can't.


---


## 6. The Extraction Pipeline


This is the hardest engineering challenge. The quality of the entire system depends on the triplet extractor.


### 6.1 Three Extraction Strategies


```mermaid
flowchart TD
   subgraph STRATEGY_A["Strategy A: Offline Pre-extraction"]
       A1["Run extraction model over<br/>entire training corpus"]
       A2["Store as parallel dataset:<br/>(triplets, raw_tokens, targets)"]
       A3["Modify dataloader.py to<br/>yield triplet context"]
       A4["Zero training speed impact"]
   end


   subgraph STRATEGY_B["Strategy B: Sidecar Model"]
       B1["Small distilled model<br/>(T5-small or similar)"]
       B2["Runs async alongside<br/>main model at inference"]
       B3["Converts evicted tokens<br/>to triplets on-the-fly"]
       B4["Adds latency but<br/>works for any input"]
   end


   subgraph STRATEGY_C["Strategy C: Self-Extraction"]
       C1["Teach the model to emit<br/>its own triplet summaries"]
       C2["New special tokens:<br/>triplet_start, triplet_end"]
       C3["Trained during SFT to<br/>periodically summarize"]
       C4["Most elegant — no<br/>external dependencies"]
   end


   STRATEGY_A -->|"Best for<br/>initial experiments"| REC["Recommended Path"]
   STRATEGY_C -->|"Best for<br/>production"| REC
```


**Strategy A** is the right starting point. It's the least risky and lets you test whether the model actually benefits from triplet context before investing in live extraction.


**Strategy C** is the most exciting long-term. nanochat already has tool-use tokens (`<|python_start|>`, `<|output_start|>` in `tokenizer.py`). Adding `<|triplet_start|>` / `<|triplet_end|>` follows the same pattern. The model learns to emit structured summaries of its own context — a form of **learned, structured self-compression**.


### 6.2 Handling the Hard Cases


```mermaid
flowchart TD
   PROBLEM1["Coreference:<br/>'The CEO resigned.<br/>She moved to Florida.'"]
   SOLUTION1["Entity linking resolves<br/>'She' → 'CEO' → 'Jane Smith'<br/>before triplet extraction"]


   PROBLEM2["Negation:<br/>'Biden did NOT sign the bill'"]
   SOLUTION2["Relation includes polarity:<br/>(Biden, did_not_sign, Bill)<br/>or (Biden, signed, Bill, neg=true)"]


   PROBLEM3["Temporal update:<br/>'Alice was promoted'<br/>overwrites<br/>'Alice is an intern'"]
   SOLUTION3["Triplet memory supports<br/>UPSERT: new triplet about<br/>same (subject, predicate)<br/>replaces the old one"]


   PROBLEM1 --> SOLUTION1
   PROBLEM2 --> SOLUTION2
   PROBLEM3 --> SOLUTION3
```


These are real challenges, but they're **solvable engineering problems**, not fundamental blockers. Coreference resolution, negation handling, and temporal updates are all active research areas with working solutions.


---


## 7. Implementation Plan for nanochat


### 7.1 Four Incremental Steps


```mermaid
flowchart LR
   STEP1["Step 1<br/>Pre-extract triplets<br/>from FineWeb data"]
   STEP2["Step 2<br/>Add TripletEncoder<br/>to gpt.py"]
   STEP3["Step 3<br/>Train with triplet<br/>context + next-token loss"]
   STEP4["Step 4<br/>Evaluate: does the<br/>model use the triplets?"]


   STEP1 --> STEP2 --> STEP3 --> STEP4
```


**Step 1 — Data preparation.** Run an off-the-shelf relation extraction model (e.g., REBEL, or a prompted LLM) over the FineWeb training data. For each document, produce a parallel file of extracted triplets. Store alongside the existing parquet shards.


**Step 2 — Model changes.** Add to `gpt.py`:
- An entity/relation vocabulary and embedding table
- A `TripletEncoder` MLP: `(e_subj, e_rel, e_obj) → d_model`
- A temporal position encoding for triplets
- Modified forward pass: prepend triplet embeddings before token embeddings
- An attention mask that lets tokens attend to triplets but not vice versa


**Step 3 — Training.** Modify `dataloader.py` to yield `(triplets, input_tokens, target_tokens)`. Modify `base_train.py` to pass triplets through the encoder and concatenate. Loss function stays **identical** — cross-entropy next-token prediction. The model learns to leverage triplet context through gradient signal alone.


**Step 4 — Validation.** Run ablations:
- Does masking out triplets increase perplexity? (If not, the model ignores them — stop here.)
- Does providing more triplets from further back improve performance?
- Compare CORE eval scores with and without triplet context.
- Compare against a baseline with equivalent raw-token context length.


### 7.2 What Changes, What Doesn't


```mermaid
graph TD
   subgraph UNCHANGED["Unchanged (95% of nanochat)"]
       U1["optim.py — Muon + AdamW"]
       U2["engine.py — Inference with KV cache"]
       U3["tokenizer.py — BPE tokenization"]
       U4["core_eval.py — CORE metric"]
       U5["flash_attention.py — FA3/SDPA"]
       U6["Loss function — Cross-entropy"]
       U7["Training loop — base_train.py"]
       U8["Scripts — SFT, RL, Web UI"]
   end


   subgraph CHANGED["Changed"]
       C1["gpt.py — Add TripletEncoder,<br/>modify forward() to prepend<br/>triplet embeddings"]
       C2["dataloader.py — Yield triplet<br/>context alongside token batches"]
       C3["New: triplet extraction<br/>preprocessing script"]
   end
```


The change is **surgical**. The transformer architecture, optimizer, loss, evaluation — all untouched. You're adding a new input pathway, not redesigning the model.


---


## 8. Why This Could Matter


### 8.1 The Scaling Argument


```mermaid
graph LR
   subgraph TODAY["Scaling today"]
       T1["More context = quadratic cost"]
       T2["128K context on H100:<br/>enormous memory + compute"]
       T3["Most of that context<br/>is syntactic filler"]
   end


   subgraph FUTURE["With triplet compression"]
       F1["Same compute budget"]
       F2["3-5x more effective context"]
       F3["Only facts survive compression"]
       F4["Model attends to<br/>information, not noise"]
   end


   TODAY -->|"Same hardware,<br/>more knowledge"| FUTURE
```


The current path to longer context is brute force: more positions, more memory, more FLOPs. Triplet compression offers an alternative scaling axis — **compress the information, not expand the compute**. A model with 2048 positions and triplet compression could carry the factual bandwidth of an 8K-context model at a fraction of the cost.


### 8.2 The Interpretability Win


Every other compression method (pooling, learned embeddings, KV cache eviction) produces opaque vectors. You cannot look at a compressed memory slot and say "this represents the fact that Alice is an engineer."


With triplets, you can. The model's memory is a readable, debuggable knowledge graph. When the model makes a mistake, you check the triplets: was the fact there? Was it extracted correctly? Was the temporal position right? This is **mechanistic interpretability for free**.


### 8.3 The Composability Win


```mermaid
flowchart LR
   subgraph CONV_A["Conversation A"]
       CA1["(Project, framework, FastAPI)"]
       CA2["(Project, database, Postgres)"]
   end


   subgraph CONV_B["Conversation B"]
       CB1["(Project, auth, JWT)"]
       CB2["(Project, deploy, AWS)"]
   end


   subgraph MERGED["Merged Context"]
       M1["(Project, framework, FastAPI)"]
       M2["(Project, database, Postgres)"]
       M3["(Project, auth, JWT)"]
       M4["(Project, deploy, AWS)"]
   end


   CONV_A --> MERGED
   CONV_B --> MERGED
```


Merging context from two conversations is a **set union on triplets**. No re-encoding, no re-embedding, no re-training. You can build persistent user profiles, project contexts, or knowledge bases that carry across sessions — something raw token context fundamentally cannot do.


---


## 9. Risks and Honest Limitations


| Risk | Severity | Mitigation |
|---|---|---|
| Triplet extraction quality is poor | **High** | Start with offline pre-extraction using best available models. Measure extraction accuracy before training. |
| Model ignores triplet context entirely | **Medium** | Ablation in Step 4 catches this early. If triplets don't reduce perplexity, stop. |
| Coreference errors fragment the graph | **Medium** | Use coreference resolution as a preprocessing step. Accept some fragmentation — the model can still learn from imperfect triplets. |
| Loss of stylistic/emotional content | **Low (by design)** | Raw token window handles this. Triplets only compress *older* context where facts matter more than style. |
| Flash Attention incompatibility | **Low** | Triplets are prepended as regular positions. The attention mask is a simple block-causal structure, fully compatible with FA3. |
| Schema drift across long contexts | **Medium** | Normalize entity and relation names during extraction. Use canonical forms. |


The biggest risk is extraction quality. If the extractor produces garbage triplets, the model learns to ignore them, and you've gained nothing. **This is why offline pre-extraction and ablation testing (Steps 1 and 4) come first** — you validate the idea cheaply before committing to architecture changes.


---


## 10. Summary


```mermaid
graph TD
   IDEA["Core Idea:<br/>Compress older context into KG triplets"]


   IDEA --> WIN1["3-5x effective context<br/>at same compute cost"]
   IDEA --> WIN2["Interpretable memory<br/>(readable triplets, not opaque vectors)"]
   IDEA --> WIN3["Composable across sessions<br/>(merge = set union)"]
   IDEA --> WIN4["Entity-preserving<br/>(no pooling blur)"]
   IDEA --> WIN5["Minimal architecture change<br/>(95% of nanochat untouched)"]


   IDEA --> RISK1["Depends on extraction quality"]
   IDEA --> RISK2["Lossy for style/emotion"]
   IDEA --> RISK3["Coreference is hard"]


   WIN1 --> VERDICT["Worth building and testing.<br/>Step 1: pre-extract, Step 2: encode,<br/>Step 3: train, Step 4: measure."]
   RISK1 --> VERDICT
```


This is not a moonshot. It's a **tractable, incremental experiment** that can be validated or killed in a single training run. The potential upside — 3-5x effective context with interpretable, composable memory — justifies the engineering investment of modifying two files and preprocessing one dataset.


The model stays a language model. It still trains on text. It still predicts tokens. It just remembers better.


---


## References


- Rae et al., "Compressive Transformers for Long-Range Sequence Modelling" (2019) — [arxiv.org/abs/1911.05507](https://arxiv.org/abs/1911.05507)
- KnowFormer: Transformers for Knowledge Graph Reasoning (2024) — [arxiv.org/html/2409.12865v1](https://arxiv.org/html/2409.12865v1)
- iHT: Pre-training Transformers for KG Completion (2023) — [arxiv.org/abs/2303.15682](https://arxiv.org/abs/2303.15682)
- TGformer: Graph Transformer for KG Embedding (IEEE, 2026) — [ieeexplore.ieee.org/document/10742302](https://ieeexplore.ieee.org/document/10742302/)
- HMT: Hierarchical Memory Transformer (NAACL 2025) — [aclanthology.org/2025.naacl-long.410](https://aclanthology.org/2025.naacl-long.410.pdf)
- MemOS: A Memory Operating System for AI (2025) — [memtensor.com.cn/files/MemOS_0707](https://statics.memtensor.com.cn/files/MemOS_0707.pdf)
- APE: Context Compression with Attention (ICLR 2025)
- Language Modeling Is Compression (ICLR 2024)
- Graphormer: Do Transformers Really Perform Bad for Graph Representation? (2021) — [arxiv.org/abs/2106.05234](https://arxiv.org/abs/2106.05234)




# GPT-Style Transformers — Engineering Revision Notes

**Convention used throughout:** batch size `B`; sequence length `T`; vocabulary size `V`; hidden size `d`; per-head key/query size `d_k`; feed-forward width `d_ff`; `h` attention heads. GPT is a **decoder-only**, causal Transformer. Learned parameters change only during training; activations are temporary values recomputed from the input. In inference, every learned parameter is frozen.

---

## 1. High-Level GPT Pipeline

```
Input text
   ↓ tokenizer
token IDs  x ∈ {0,…,V−1}^(B×T)
   ↓ embedding lookup
token embeddings E[x] ∈ R^(B×T×d)
   + positional embeddings P
hidden states H⁰ ∈ R^(B×T×d)
   ↓ Transformer layer 1
H¹  ↓ Transformer layer 2  ↓ … ↓ Transformer layer L
final hidden state of last position h_last ∈ R^d
   ↓ LM head
logits z ∈ R^V  → softmax → next-token distribution p ∈ R^V
   ↓ select token → append it → repeat
```

- **What:** A function that maps a token prefix to a distribution over the next token.
- **Why:** Text generation is decomposed into a tractable repeated task: “given all previous tokens, predict one more.”
- **How:** Transform every token representation through `L` causal layers; read only the final position to predict the continuation.
- **Next:** The sampled/selected ID is appended to the prefix, so it becomes context for the next pass.

## 2. Training vs Inference

- **Training:** feed a known sequence, predict each next-token target, compute cross-entropy loss, backpropagate, update parameters.
- **Inference:** feed a prefix, compute only the next-token distribution, select an ID, append it. No loss, no gradients, no parameter updates.
- **Key distinction:** parameters are persistent learned tensors; activations are input-dependent temporary tensors.

| Component | Training | Inference |
|---|---|---|
| Embedding matrix `E ∈ R^(V×d)` | learned/updated | frozen; rows looked up |
| Positional embeddings `P ∈ R^(T_max×d)` | learned/updated (if learned) | frozen; rows looked up |
| `WQ, WK, WV ∈ R^(d×d_k)` per head | learned/updated | frozen; multiply activations |
| `WO ∈ R^(h·d_k×d)` | learned/updated | frozen |
| Feed-forward weights | learned/updated | frozen |
| Feed-forward biases | learned/updated | frozen |
| LayerNorm `γ, β ∈ R^d` | learned/updated | frozen |
| LM head `WLM ∈ R^(d×V)` | learned/updated | frozen |
| Biases, if present | learned/updated | frozen |
| Hidden states | computed; retained for gradients | computed; discarded after use (except needed cache) |
| Queries / keys / values | computed activations | computed activations; K/V may be cached |
| Attention scores / weights / outputs | computed activations | computed activations |
| Logits / probabilities | computed for loss | computed for selection |

## 3. Vocabulary

- **Vocabulary:** fixed table of `V` token pieces known by the model.
- **Vocabulary size `V`:** number of possible output classes and rows in `E`; e.g. 50k–200k.
- **Tokens:** subword pieces, words, punctuation, whitespace fragments, or bytes—not necessarily words.
- **Token ID:** integer index into the vocabulary; it has no numerical meaning or distance relation.
- **Why tokens, not words:** open-ended language makes a word-only vocabulary too large and cannot cleanly cover rare words, spelling, code, or multilingual text. Subword/byte tokenization gives finite coverage and reusable pieces.

## 4. Tokenization

```
"The cat sat"  →  ["The", " cat", " sat"]  →  [42, 317, 901]
```

- **What:** deterministic mapping between text and token IDs (typically BPE/Unigram/byte-based).
- **Why:** neural networks consume numbers; a compact reusable vocabulary prevents one class per possible word.
- **How:** tokenizer chooses vocabulary pieces that concatenate back to the original text; decoder reverses IDs to text.
- **Next:** IDs index rows of the embedding matrix.

## 5. Embedding Matrix

- **Embedding matrix `E ∈ R^(V×d)`:** learned parameter; row `E[i]` is the `d`-dimensional representation for token ID `i`.
- **Input embeddings `X = E[x] ∈ R^(B×T×d)`:** activation formed by selecting rows. `E` is one shared table; `X` varies with the prompt.
- **Why embeddings exist:** one-hot `e_i ∈ R^V` is sparse, huge, and cannot directly encode learned similarity. `e_iᵀE = E[i]` turns a discrete symbol into a dense working vector.
- **Inference:** `E` stays fixed. Input embeddings change only because different token IDs select different fixed rows.
- **Next:** add positional information to make `H⁰`.

## 6. Positional Encoding

- **Position embedding `P[t] ∈ R^d`:** learned parameter row (or a fixed sinusoidal/rotary-derived positional signal) for position `t`.
- **Initial hidden state `H⁰[:,t,:] = X[:,t,:] + P[t] ∈ R^d`:** activation.
- **Why:** self-attention alone is permutation-equivariant: swapping token positions swaps outputs, with no inherent “first/after/before” concept. Language order must be supplied.
- **Embedding answers:** “which token is this?” Positional signal answers: “where is it / how far apart are tokens?”
- **Next:** `H⁰` is the first layer’s input.

## 7. Hidden States

- **Hidden state `Hˡ ∈ R^(B×T×d)`:** activation after layer `l`; one `d`-vector per position. `H⁰` is embedding + position.
- **What it represents:** the model’s current contextual interpretation of each token, given only allowed earlier tokens.
- **Why it exists:** token IDs are discrete labels. Every layer needs a continuous workspace where context can be read, transformed, and preserved.
- **Embedding vs hidden state:** embedding is context-free lookup; hidden state is context-dependent and layer-specific.
- **Attention output vs hidden state:** attention output is only the context-read contribution of one sublayer. After projection, residual, normalization, and FFN, it becomes part of the new hidden state.
- **Why richer per layer:** earlier layers often form local/syntactic features; later layers can compose features already mixed across context. No single layer is required to have a human-readable meaning.
- **Invariant:** hidden states are the **only** per-token vectors passed from one Transformer layer to the next. Q/K/V, scores, and attention outputs are intermediate activations and disappear after that layer (except cached K/V at inference).

## 8. Complete Transformer Layer

Post-LayerNorm form, matching this note’s worked example (many modern GPTs use **pre-LN**: normalize before each sublayer instead):

```
Hˡ⁻¹ → Q,K,V → causal multi-head attention → WO → + Hˡ⁻¹ → LayerNorm
    → Feed Forward → + previous result → LayerNorm → Hˡ
```

- **Why two sublayers:** attention communicates across positions; FFN computes a nonlinear per-position transformation.
- **Next:** `Hˡ` feeds the next identical-structure layer with different parameters.

## 9. `WQ`, `WK`, `WV`, `WO`

For one head, with input `H ∈ R^(B×T×d)`:

| Tensor | Shape | Type | Purpose / next use |
|---|---:|---|---|
| `WQ` | `d×d_k` | learned | makes queries `Q = HWQ` |
| `WK` | `d×d_k` | learned | makes keys `K = HWK` |
| `WV` | `d×d_v` | learned | makes values `V = HWV` (usually `d_v=d_k`) |
| `WO` | `(h·d_v)×d` | learned | mixes concatenated head outputs back to hidden width |

- **Why three projections:** “where to look” (Q), “how to be found” (K), and “what content to send” (V) are different jobs. Forcing one vector to do all three reduces expressive freedom.
- **Why each layer differs:** each layer performs a new learned computation over a differently contextualized `H`; sharing would repeat the same operation.
- **`WO` vs `WLM`:** `WO` maps attention features back into hidden space `d`; LM head maps final hidden space to vocabulary `V`.

## 10. Query, Key, Value

For a head: `Q,K ∈ R^(B×T×d_k)`, `V ∈ R^(B×T×d_v)` are temporary activations.

- **Q:** “What information am I looking for?” A query at destination position `i` is compared with all allowed keys.
- **K:** “What kind of information do I contain?” A key at source position `j` determines whether `j` matches a query.
- **V:** “What information will I contribute if selected?”
- **Why compare Q with K:** dot products give learned compatibility, separately for every destination/source pair.
- **Why never compare V:** a value is payload, not addressing metadata. It is mixed only after addressing weights are decided.

## 11. Attention

For one head and one batch item, `Q,K ∈ R^(T×d_k)`, `V ∈ R^(T×d_v)`:

```
QKᵀ → raw scores S ∈ R^(T×T) → add causal mask → divide by √d_k
   → row-wise softmax A ∈ R^(T×T) → AV → O ∈ R^(T×d_v)
```

`S_ij = q_i · k_j`. Position `i` is the query/destination; `j` is key/value/source.

- **Dot product / similarity:** high when learned query and key features align; “similarity” means learned relevance, not necessarily semantic similarity.
- **Scores `S`:** unnormalized compatibility numbers; can be negative and do not sum to one.
- **Weights `A`:** nonnegative normalized scores; each row sums to 1 over allowed sources.
- **Output `O_i = Σ_j A_ij v_j`:** context vector gathered for position `i`.
- **Next:** concatenate head outputs, project with `WO`, then residual path.

## 12. Causal Mask

- **Mask `M ∈ R^(T×T)`:** fixed non-learned tensor, `M_ij=0` when `j≤i`, `−∞` when `j>i`.
- **Why:** GPT is decoder-only and must predict token `i+1` using positions `≤i`. Allowing future positions during training leaks the answer (“cheating”) and breaks generation.
- **Applied:** `S_masked = S + M` before softmax; `softmax(−∞)=0`.
- **Next:** each row can weight only its past and itself.

## 13. Scaling

`S_scaled = S_masked / √d_k`.

- **Why:** if Q/K components have roughly unit variance, a dot product’s variance grows with `d_k`. Larger raw scores make softmax excessively peaked, producing tiny gradients and brittle early training.
- **Intuition:** divide by expected score scale so attention sharpness does not accidentally depend on head width.

## 14. Softmax

`softmax(s)_j = exp(s_j) / Σ_r exp(s_r)`.

- **Why probabilities/weights:** a convex mixture gives a stable differentiable routing mechanism and comparable relative allocation across sources.
- **Why raw scores cannot be used directly:** they are unbounded, signed, and not normalized; they would arbitrarily scale or subtract values.
- **Scores:** preference evidence. **Weights:** normalized allocation. Softmax is applied independently for each query row.

## 15. Weighted Sum of Values

`O = AV`, so `o_i = Σ_{j≤i} a_ij v_j`.

- **What:** matrix multiplication blending each allowed source’s value according to that destination’s attention weights.
- **Why blend, not copy:** relevance is usually distributed—syntax, entity attributes, and prior constraints can come from multiple places. A weighted mixture is differentiable and supports uncertainty.
- **Next:** one output vector per position/head is projected and merged into the residual stream.

## 16. Attention Output

- **Attention output `O ∈ R^(B×T×d_v)` per head:** activation containing information read from the context for each position.
- **What it is not:** not the entire representation of a token; it omits the direct residual input and the FFN update.
- **Hidden state:** persistent layer-to-layer residual-stream vector. **Attention output:** ephemeral retrieval result inside one layer.

## 17. `WO`

- **Output projection `WO ∈ R^(h·d_v×d)`:** learned parameter. `U = Concat(O¹,…,Oʰ)WO ∈ R^(B×T×d)`.
- **Why:** heads use small specialist subspaces. Concatenation must be mixed and returned to the residual stream’s common width `d`.
- **Multi-head intuition:** distinct heads can learn different relation types or routing patterns in parallel; not every head necessarily has an interpretable role.
- **Not LM head:** `WO` produces hidden-space updates for every token; `WLM` produces vocabulary scores from the final layer.

## 18. Residual Connections

`R = H + U` (and later `R₂ = N + F`). Shapes match: all are `B×T×d`.

- **Why add instead of replace:** preserve the current representation while letting a sublayer contribute an incremental correction.
- **Advantages:** direct gradient paths, easier optimization of deep stacks, and the option for a layer to learn a small/no-op update.
- **Next:** the sum is normalized before/after the following computation depending on architecture.

## 19. LayerNorm

For one vector `r ∈ R^d`: `μ = (1/d)Σ_k r_k`, `σ²=(1/d)Σ_k(r_k−μ)²`; `LN(r)=γ ⊙ (r−μ)/√(σ²+ε)+β`.

- **`γ, β ∈ R^d`:** learned parameters; `μ, σ²` are temporary per-token statistics; `ε` is fixed numerical stabilizer.
- **Why:** keeps residual-stream feature scale controlled, improves conditioning and training stability, while `γ/β` let the model restore useful per-feature scale/offset.
- **Unlike BatchNorm:** normalizes across features of each token independently; sequence/batch length need not be fixed.

## 20. Feed Forward Network

For each token independently: `F(H)=W₂ φ(HW₁+b₁)+b₂`.

| Tensor | Shape | Type |
|---|---:|---|
| `W₁` | `d×d_ff` | learned |
| `b₁` | `d_ff` | learned (if present) |
| `W₂` | `d_ff×d` | learned |
| `b₂` | `d` | learned (if present) |

- **Why expand then compress:** `d_ff` (often ~4d) gives a larger feature workspace; compression writes the selected nonlinear features back to `d`.
- **Why nonlinear activation:** stacked linear maps collapse into one linear map. `ReLU`, `GELU`, or gated `SwiGLU` create conditional, richer transformations.
- **Attention vs FFN:** attention mixes information **between positions**; FFN transforms features **at each position**, using identical weights at all positions.
- **Next:** add its result to the attention-residual stream and normalize to form a new hidden state.

## 21. End of Layer

`Hˡ = LN(R + F(R)) ∈ R^(B×T×d)` in this post-LN schematic.

- **What:** new hidden states, one contextual vector per token.
- **Why:** combines preserved information, retrieved context, and nonlinear local computation.
- **Next:** exactly `Hˡ`—not Q/K/V or weights—feeds layer `l+1`.

## 22. Layer 2

- **What:** same architecture and shapes as layer 1.
- **Why:** depth permits iterative contextual computation: later Q/K/V are formed from already contextual hidden states.
- **Crucial:** it has independently learned `WQ², WK², WV², WO²`, FFN weights, and LayerNorm parameters. Same blueprint ≠ same function.

## 23. LM Head (`WLM`)

- **LM head / output projection `WLM ∈ R^(d×V)`:** learned parameter mapping final last-token state `h_last ∈ R^d` to `z=h_last WLM+bLM ∈ R^V`; `bLM ∈ R^V` is optional learned bias.
- **Why:** generation chooses among `V` discrete token classes, so hidden dimension must become vocabulary dimension.
- **`WO` vs `WLM`:** `WO: h·d_v → d`, inside every layer, preserves working space. `WLM: d → V`, once at the end, produces class scores.
- **Weight tying:** many models set `WLM = Eᵀ` (orientation-dependent notation), sharing input and output token representations. This reduces parameters and often improves learning; models may also leave them untied.

## 24. Logits

- **Logits `z ∈ R^V`:** raw, unnormalized output scores—one per vocabulary token.
- **Why:** linear class scores retain signed confidence evidence and are convenient for training; softmax converts them to a normalized distribution only when needed.
- **Logits vs probabilities:** logits need not be positive or sum to one; probabilities are in `[0,1]` and sum to one.

## 25. Final Softmax

`p_i = exp(z_i)/Σ_{r=1}^V exp(z_r)`.

- **Why:** turns `V` competing logits into a categorical next-token distribution for loss calculation or sampling.
- **Next:** a decoding rule chooses one token ID.

## 26. Token Selection

- **Greedy:** choose `argmax_i p_i`; deterministic, can be repetitive.
- **Sampling:** draw from `p`; adds variety.
- **Temperature `τ`:** use `softmax(z/τ)`; lower `<1` sharpens, higher `>1` flattens.
- **Top-k:** retain only the `k` highest-logit tokens, renormalize.
- **Top-p (nucleus):** retain smallest set with cumulative probability ≥ `p`, renormalize.
- **Why:** model distribution is not itself a complete product decision; decoding controls the quality/diversity trade-off.

## 27. Autoregressive Generation

```
prefix → all hidden states → last-position state → next-token distribution
       → choose ID → append to prefix → repeat
```

- **Why only last hidden state predicts:** after prefix `[x₁,…,x_T]`, position `T` has access to the complete prefix and is trained to predict `x_{T+1}`.
- **Why every token is still updated:** last state needs context; attention builds that context from updated states at all earlier positions. During training, every position supplies a parallel next-token prediction.

## 28. KV Cache

- **Without cache:** generating each new token recomputes K and V for every old token in every layer—wasteful.
- **With cache:** for layer `l`, store old `K_cacheˡ,V_cacheˡ ∈ R^(B×T_old×d_k/d_v)`. On one new token, compute only `q_new,k_new,v_new`; append K/V to cache; `q_new` attends over cached + new keys/values.
- **Why only new Q:** old positions cannot see the future under a causal mask, so their old outputs do not change when a token is appended. The new position is the only one needing a new context read.
- **Cache contains:** per-layer keys and values, not attention weights and not usually queries.

## 29. Complete Worked Example

**Deliberately tiny illustrative model:** `V=5`, `d=d_k=d_v=2`, `T=3`, one head, two layers, no biases, post-LN, `ε=0`, FFN is `ReLU` with `d_ff=2`. This makes arithmetic visible, not realistic. Vocabulary: `[The:0, cat:1, sat:2, on:3, mat:4]`; input IDs `[0,1,2]` = “The cat sat”.

`E ∈ R^(5×2) = [[1,0],[0,1],[1,1],[−1,0],[0,−1]]`; `P ∈ R^(3×2) = [[0,0],[0,0],[0,0]]`.

`H⁰=E[x]+P = [[1,0],[0,1],[1,1]]` (rows: The, cat, sat).

### Layer 1

Parameters: `WQ¹=WK¹=WV¹=WO¹=I₂`; `W₁¹=I₂`; `W₂¹=I₂`; LayerNorm `γ=[1,1], β=[0,0]`.

**Q/K/V (each `3×2` activation):**

`Q¹=H⁰WQ¹=[[1,0],[0,1],[1,1]] [[1,0],[0,1]]=[[1,0],[0,1],[1,1]]`.

`K¹=H⁰WK¹=[[1,0],[0,1],[1,1]]`; `V¹=H⁰WV¹=[[1,0],[0,1],[1,1]]`.

**Scores:** `S¹=Q¹(K¹)ᵀ = [[1,0,1],[0,1,1],[1,1,2]]`.

**Mask then scale (`√d_k=√2≈1.414`):**

`S_masked¹=[[1,−∞,−∞],[0,1,−∞],[1,1,2]]`;
`S_scaled¹≈[[0.707,−∞,−∞],[0,0.707,−∞],[0.707,0.707,1.414]]`.

**Row softmax:** `A¹≈[[1,0,0],[0.330,0.670,0],[0.248,0.248,0.503]]`.

**Weighted values:**

`O¹=A¹V¹ ≈ [[1,0], [0.330·[1,0]+0.670·[0,1]], [0.248·[1,0]+0.248·[0,1]+0.503·[1,1]]]`

`O¹≈[[1,0],[0.330,0.670],[0.751,0.751]]`.

**`WO`, residual, LN:** `U¹=O¹WO¹=O¹`; `R¹=H⁰+U¹≈[[2,0],[0.330,1.670],[1.751,1.751]]`.

For 2 features, LN maps any unequal `[a,b]` to `[1,−1]` if `a>b`, and `[-1,1]` if `a<b`; equal features map to `[0,0]` (with the stated `ε=0` convention understood as limiting case). Thus `N¹=LN(R¹)=[[1,−1],[-1,1],[0,0]]`.

**FFN:** `G¹=N¹W₁¹=N¹`; `ReLU(G¹)=[[1,0],[0,1],[0,0]]`; `F¹=ReLU(G¹)W₂¹=[[1,0],[0,1],[0,0]]`.

**Final residual + LN:** `Rff¹=N¹+F¹=[[2,−1],[-1,2],[0,0]]`; `H¹=LN(Rff¹)=[[1,−1],[-1,1],[0,0]]`.

### Layer 2

Parameters again are identity matrices and `γ=[1,1],β=[0,0]` only to keep the worked arithmetic compact; in a real model these are separate learned tensors.

`Q²=K²=V²=H¹=[[1,−1],[-1,1],[0,0]]`.

`S²=Q²(K²)ᵀ=[[2,−2,0],[-2,2,0],[0,0,0]]`.

`S_scaled²≈[[1.414,−∞,−∞],[-1.414,1.414,−∞],[0,0,0]]` after causal masking/scaling.

`A²≈[[1,0,0],[0.056,0.944,0],[0.333,0.333,0.333]]`.

`O²=A²V²≈[[1,−1],[0.056[1,−1]+0.944[−1,1]],[0,0]]≈[[1,−1],[−0.888,0.888],[0,0]]`.

`U²=O²`; `R²=H¹+U²≈[[2,−2],[−1.888,1.888],[0,0]]`; `N²=LN(R²)=[[1,−1],[-1,1],[0,0]]`.

`F²=ReLU(N²)=[[1,0],[0,1],[0,0]]`; `H²=LN(N²+F²)=[[1,−1],[-1,1],[0,0]]`.

**LM head and prediction:** last state `h_last=[0,0]`. Let
`WLM ∈ R^(2×5)=[[1,0,−1,0,1],[0,1,0,−1,−1]]`.

`z=h_last WLM=[0,0]WLM=[0,0,0,0,0]`; `softmax(z)=[0.2,0.2,0.2,0.2,0.2]`.

This toy’s final state is deliberately degenerate because of symmetric choices. Greedy has a five-way tie; choose a tie rule or sample. **Engineering lesson:** all calculations follow the real pipeline; trained non-symmetric weights create useful logits. If `h_last=[1,−1]` instead, `z=[1,−1,−1,1,2]`, softmax ≈ `[0.207,0.028,0.028,0.207,0.563]`, predicting `mat` greedily.

## 30. Common Beginner Confusions

| Question | Revision answer |
|---|---|
| Embedding matrix vs input embeddings | `E` is learned `V×d` table; input embeddings are selected `T×d` rows. |
| Embedding vs hidden state | Lookup is context-free; hidden state is contextual layer activation. |
| Hidden state vs attention output | Hidden state survives layers; attention output is one temporary retrieval contribution. |
| Scores vs weights | Scores are raw Q·K compatibility; weights are masked/scaled softmax scores. |
| Weights vs output | Weights say where to read; output is weighted values—the content read. |
| Query vs key | Query requests; key advertises matchability. |
| Key vs value | Key chooses relevance; value supplies payload. |
| `WO` vs `WLM` | `WO` returns attention to hidden width; `WLM` maps hidden width to vocabulary. |
| FFN vs attention | FFN mixes features within one position; attention mixes information across positions. |
| Training vs inference | Training updates parameters using targets; inference freezes them and generates. |
| Parameters vs activations | Parameters persist/learn; activations are computed per prompt. |
| Layer vs attention head | Layer is full attention+FFN block; head is one parallel attention subspace inside it. |
| Attention layer vs Transformer layer | Attention is a sublayer; Transformer layer also includes residuals, norms, FFN. |
| Logits vs probabilities | Logits are raw class scores; probabilities are normalized softmax values. |
| Attention softmax vs LM softmax | Attention normalizes over source positions per query; LM softmax over vocabulary tokens. |
| Why only last token predicts | It has the full prefix and corresponds to the next unfilled position. |
| Why every token updates | Last token’s context depends on representations of earlier tokens. |
| Why every layer has different `WQ` | Each layer needs a new learned relation test over richer states. |
| Why Q/K/V disappear | They are layer-local routing activations; only `Hˡ` is next layer’s state. |
| What is passed next | New hidden states `Hˡ`, plus cached K/V only during incremental inference. |
| Why residuals | Preserve signal and make deep optimization feasible. |
| Why LayerNorm | Stabilize feature scale and optimization. |
| Why FFN | Add nonlinear per-token feature computation. |
| Why output projection | Merge heads back to the shared residual stream. |
| Why LM head | Convert `d` features into `V` token choices. |
| Vocabulary size vs embedding size | `V` = number of token types/classes; `d` = vector feature width. |
| Hidden vs vocabulary dimension | Hidden dimension is internal workspace; vocabulary dimension is final number of possible tokens. |

## 31. One-Page Summary

```
Text → tokenizer → IDs x (B×T)
     → E[x] + P = H⁰ (B×T×d)
     → repeat L times:
        Q=HWQ, K=HWK, V=HWV
        A=softmax((QKᵀ + causal mask)/√d_k)
        U=Concat(AV across heads)WO
        N=LN(H+U)
        H=LN(N + W₂φ(NW₁+b₁)+b₂)
     → h_last = H[:,T,:] (d)
     → logits z=h_last WLM+bLM (V)
     → p=softmax(z) (V)
     → select token ID → append → repeat
```

- `E,P,WQ,WK,WV,WO,W₁,W₂,γ,β,WLM` (and biases if used) are learned parameters: update in training, frozen in inference.
- `H,Q,K,V,S,A,O,U,N,z,p` are activations: recomputed from the prompt. K/V can be retained in an inference KV cache.
- Attention answers **where should this position read?**; values provide **what it reads**; FFN answers **how should this position transform its features?**
- Causal masking enforces: position `i` sees only `≤i`. The final position therefore predicts the next token without seeing it.

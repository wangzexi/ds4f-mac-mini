# Formal Decode flow

This is the production Decode path for the fixed DeepSeek-V4-Flash-0731
runtime on the M4 Mini.  Prefill is deliberately outside this flow.

```mermaid
flowchart TD
    A[Request arrives] --> B[Reuse or build KV prefix]
    B --> C[Prefill completes]
    C --> D{Uncached Prefill rows > 18?}
    D -->|Yes| E[Select global LRU]
    D -->|No| F[Select probation-LRU\nand protect previous route]
    E --> G[Decode one token]
    F --> G

    G --> H[For each transformer layer 0..42]
    H --> I[Attention reads current hidden\nand existing KV]
    I --> J[Router selects six experts]
    J --> K{Selected expert in\nglobal cache?}
    K -->|Hit| L[Bind resident Metal buffers]
    K -->|Miss| M[Choose global victim]
    M --> N[Protect in-flight/current\nand required hash entries]
    N --> O[SSD pread selected expert]
    O --> P[Install expert into shared cache]
    P --> L
    L --> Q[Run six-expert MLP\nand weighted combine]
    Q --> R[Update route heat and cache clocks]
    R --> S{More layers?}
    S -->|Yes| H
    S -->|No| T[Final logits and greedy next token]
    T --> U{Stop or max tokens?}
    U -->|No| G
    U -->|Yes| V[Persist KV and finish response]
```

## Production cache rule

The Decode expert cache is one global pool of about 967 entries.  There is no
per-layer fixed reservation in production.  A miss can evict an eligible entry
from any layer, subject to these protections:

1. in-flight entries;
2. the current layer's selected experts;
3. the required hash-layer protection;
4. the previous route during the short probation-LRU regime.

The cache policy only chooses the victim.  It does not change Router results,
the six selected expert IDs, the weighted expert combine, or greedy sampling.

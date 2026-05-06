# CuTe → GPT roadmap

## Foundations — layouts on paper
1. Layout basics — shape, stride, offset ✅
2. Hierarchical layouts — nested shapes for tiles-of-tiles ✅
3. Tiling — `cute.tiled_divide` derives the hierarchy ✅

## First kernels
4. Vector add — one thread per element ✅
5. Bounds checking — `cute.ceil_div` + `if i < L` for arbitrary sizes

## Memory hierarchy
6. 2D thread-block tiling with `cute.local_tile` — each block owns a tile
7. Shared memory: global → smem → global, `cp.async`
8. Thread-level partitioning (`local_partition`) — divide a tile across threads
9. Vectorized loads — the `assumed_align=16` payoff, LD.128 in practice

## The matmul ladder
10. Naive GEMM — outer-product accumulation in registers
11. Tiled GEMM with smem staging
12. TensorCore GEMM via `cute.make_tiled_mma` — Hopper/Blackwell MMA atoms
13. Pipelined GEMM — overlap load and compute, multi-stage smem

## Transformer building blocks
14. Bias + GELU/SiLU epilogue fused into GEMM
15. RMSNorm / LayerNorm — reductions
16. Softmax with online (Welford-style) max+sum
17. RoPE

## Attention, three steps
18. Materialized attention — S = QKᵀ in HBM, softmax kernel, S·V. *Feel the memory cost.*
19. Tiled attention, offline softmax — one fused kernel, tile loop, full-row softmax per Q-tile. *Lesson: fusion.*
20. FlashAttention — swap offline for online softmax. *Lesson: streaming K.*

## Putting it together
21. Transformer block — attention + MLP + residuals + norms
22. Stack N blocks, embeddings, LM head → GPT forward pass
23. KV cache for autoregressive decode
24. (Stretch) backward pass, or stop at inference

---

Big cliffs: **10 → 13** (tiled-MMA clicking), and **18 → 20** (attention as fused, streaming GEMMs). Everything else is plumbing on top of those two ideas.

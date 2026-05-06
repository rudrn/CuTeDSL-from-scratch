"""
Puzzle #5 — Bounds checking: kernels for sizes that DON'T divide evenly.

================================================================
THE GOAL
================================================================

In puzzle #4 we chose M=N=1024 and threads_per_block=256, which
divides perfectly: 1024*1024 / 256 = 4096 blocks, no remainder.

Real life is not that polite. What if N=1000? Or the user passes
a vector of length 12345? You can't launch 12345/256 = 48.22 blocks.

The fix is universal and shows up in EVERY CuTe kernel from now on:

    1) launch CEIL(total / block_size) blocks  (one extra, partial)
    2) inside the kernel: `if i < total:` before touching memory

The "extra" block has some threads with i >= total. They must do
nothing — otherwise they read/write past the tensor. Out-of-bounds
writes on the GPU don't raise; they silently corrupt random memory
or segfault later in something completely unrelated. Hard to debug.

================================================================
SHAPE CHANGE: 1D this time
================================================================

Puzzle 4 used a 2D (M, N) tensor and divided i into (row, col).
For this puzzle we use a flat 1D vector of length L. Less arithmetic,
more focus on the predicate.

       a = torch.randn(L, device="cuda", dtype=torch.float16)

Each thread handles ONE element at index `i = bidx * bdim + tidx`.
If i >= L, do nothing.

================================================================
CEIL DIVISION
================================================================

cute exposes `cute.ceil_div(a, b)` — equivalent to (a + b - 1) // b.
That gives "smallest number of blocks that COVERS every element,
possibly with leftover threads in the last block."

Example: L=1000, threads_per_block=256
    ceil_div(1000, 256) = 4
    total threads launched = 4 * 256 = 1024
    threads with i in [0, 1000)  -> do work
    threads with i in [1000, 1024) -> skip (24 idle threads)

================================================================
PREDICT
================================================================

Q1. With L=1000, threads_per_block=256, how many blocks?
    your guess: ???

Q2. How many threads launch in total?
    your guess: ???

Q3. How many of them are "idle" (i >= L)?
    your guess: ???

Q4. If we FORGOT the `if i < L` check and L=1000, what's the
    worst that could happen?
    your guess: ???

================================================================
"""

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


################################################################
# YOUR JOB: fill in the TODOs.
# Use puzzle #4 as a reference — the structure is identical
# except for the predicate and the 1D shape.
################################################################


@cute.kernel
def add_kernel_1d(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    L: cutlass.Constexpr,   # total length, baked into the kernel at compile time
):
    # TODO 1: who am I? (same three lines as puzzle 4)
    tidx, _, _ = ???
    bidx, _, _ = ???
    bdim, _, _ = ???

    # TODO 2: my unique global id.
    i = ???

    # TODO 3: BOUNDS CHECK. Only do work if i is a valid index.
    #         Hint: a plain Python `if` works inside a @cute.kernel.
    if ???:
        gC[i] = gA[i] + gB[i]


@cute.jit
def add_1d(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, L: cutlass.Constexpr) -> None:
    threads_per_block: int = 256

    # TODO 4: compute number of blocks with CEIL division so every
    #         element is covered (last block may be partially idle).
    #         Use cute.ceil_div.
    num_blocks = ???

    add_kernel_1d(mA, mB, mC, L).launch(
        grid=(num_blocks, 1, 1),
        block=(threads_per_block, 1, 1),
    )


def run(L: int) -> None:
    print(f"\n--- L = {L} ---")
    a = torch.randn(L, device="cuda", dtype=torch.float16)
    b = torch.randn(L, device="cuda", dtype=torch.float16)
    c = torch.zeros(L, device="cuda", dtype=torch.float16)

    a_ = from_dlpack(a, assumed_align=16)
    b_ = from_dlpack(b, assumed_align=16)
    c_ = from_dlpack(c, assumed_align=16)

    compiled = cute.compile(add_1d, a_, b_, c_, L)
    compiled(a_, b_, c_, L)

    torch.testing.assert_close(c, a + b)
    print(f"PASS  (L={L})")


def main() -> None:
    # The whole point: try sizes that DON'T divide cleanly by 256.
    for L in [256, 1000, 1, 257, 12345]:
        run(L)


if __name__ == "__main__":
    main()

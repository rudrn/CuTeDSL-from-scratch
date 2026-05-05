"""
Puzzle #4 — Your first CuTe kernel: vector add on the GPU.

================================================================
WHAT YOU'RE BUILDING
================================================================

A naive elementwise C = A + B kernel, with one thread per element.
Embarrassingly parallel — but you'll see the WHOLE scaffolding
that EVERY CuTe kernel uses. Once this works, more complex
kernels are just "more interesting math inside the kernel."

================================================================
THE THREE INGREDIENTS
================================================================

1) DATA — torch tensors on the GPU. CuTe doesn't allocate;
   we hand it pointers via DLPack.

       a = torch.randn(M, N, device="cuda", dtype=torch.float16)
       a_ = from_dlpack(a, assumed_align=16)
                       # ^ wraps a's pointer as a CuTe tensor

2) KERNEL — code each thread runs. Decorated @cute.kernel.
   Inside, ask "who am I?" via cute.arch.thread_idx() etc.,
   then do your work.

3) HOST FUNCTION — decorated @cute.jit. Computes a launch
   configuration (grid + block) and calls kernel.launch(...).

The full pipeline:
       cute.compile(host_fn, *cute_tensors)  -> compiled callable
       compiled(*cute_tensors)               -> runs on GPU
       torch.testing.assert_close(c, a + b)  -> sanity check

================================================================
THE KERNEL — annotated
================================================================

@cute.kernel
def add_kernel(gA, gB, gC):
    tidx, _, _ = cute.arch.thread_idx()   # which thread inside this block
    bidx, _, _ = cute.arch.block_idx()    # which block inside the grid
    bdim, _, _ = cute.arch.block_dim()    # how many threads per block

    # Global thread id — uniquely names this thread among ALL threads in the launch.
    i = bidx * bdim + tidx

    # Map flat id -> 2D (row, col).
    m, n = gA.shape
    row = i // n
    col = i %  n

    # Load, compute, store. The tensor `gA` knows its layout, so
    # gA[row, col] does the address math for us.
    gC[row, col] = gA[row, col] + gB[row, col]

================================================================
LAUNCH MATH (don't skip — this trips up everyone the first time)
================================================================

Total elements   = M * N
Threads / block  = 256
Blocks needed    = (M * N) / 256       (assuming it divides evenly)

If M=1024, N=1024:
       total = 1,048,576 elements
       blocks = 1,048,576 / 256 = 4096
       launch: grid=(4096,1,1) block=(256,1,1)
       => 4096 * 256 = 1,048,576 threads — one per element.

================================================================
PREDICT
================================================================

Q1. With M=1024, N=1024, threads_per_block=256, how many
    THREAD BLOCKS does the launch create?
    your guess:

Q2. If thread tidx=0 is in block bidx=0, what (row, col) does it
    process?
    your guess:

Q3. If thread tidx=5 is in block bidx=3, what global element
    index does it touch? (i.e. row*N + col)
    your guess:

Q4. We pick threads_per_block=256. Why is 256 (and not, say, 100)?
    your guess:

================================================================
"""

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


################################################################
# YOUR JOB: fill in the four TODOs below.
# Each `???` or `# TODO` line is something you have to write.
# Refer to the kernel example in the docstring above as a guide.
################################################################


@cute.kernel
def add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
):
    # TODO 1: get this thread's id within its block, the block's id in the grid,
    #         and the block size. Use cute.arch.thread_idx(), block_idx(), block_dim().
    #         Each returns a 3-tuple (x, y, z) — we only need x.
    tidx, _, _ = ???
    bidx, _, _ = ???
    bdim, _, _ = ???

    # TODO 2: compute a global thread id `i` so that every thread has a UNIQUE i.
    #         Formula: i = (which block) * (threads per block) + (which thread in block)
    i = ???

    # TODO 3: convert flat index i to a 2D (row, col) coordinate.
    #         Hint: gA.shape gives (m, n). Use // and % .
    m, n = gA.shape
    row = ???
    col = ???

    # TODO 4: do the actual work — write A[row,col] + B[row,col] into C[row,col].
    gC[row, col] = ???


@cute.jit
def add(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    threads_per_block = 256
    m, n = mA.shape

    # TODO 5: compute number of blocks needed so that
    #         num_blocks * threads_per_block == m * n.
    num_blocks = ???

    add_kernel(mA, mB, mC).launch(
        grid=(num_blocks, 1, 1),
        block=(threads_per_block, 1, 1),
    )


def main():
    M, N = 1024, 1024  # 1M elements — small enough to be quick, big enough to be real

    a = torch.randn(M, N, device="cuda", dtype=torch.float16)
    b = torch.randn(M, N, device="cuda", dtype=torch.float16)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float16)

    a_ = from_dlpack(a, assumed_align=16)
    b_ = from_dlpack(b, assumed_align=16)
    c_ = from_dlpack(c, assumed_align=16)

    print("Compiling kernel...")
    add_compiled = cute.compile(add, a_, b_, c_)
    print("Running on GPU...")
    add_compiled(a_, b_, c_)

    print("Verifying against torch...")
    torch.testing.assert_close(c, a + b)
    print("✅ PASS — your CuTe kernel matches torch.")

    # Tiny visible sample so you can see real numbers
    print("\nFirst 4x4 of c:")
    print(c[:4, :4].cpu())


if __name__ == "__main__":
    main()

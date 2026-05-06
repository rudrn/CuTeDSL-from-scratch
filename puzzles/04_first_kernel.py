"""
Puzzle #4 — Your first CuTe kernel: vector add on the GPU.

================================================================
THE GOAL
================================================================

Compute C = A + B on the GPU, one thread per element.
Simple math, but you'll wire up every piece a real CuTe kernel needs.

================================================================
HOW A CuTe KERNEL IS PUT TOGETHER
================================================================

Three pieces talk to each other:

1) TENSORS on the GPU.
   torch allocates them; CuTe wraps the pointer with from_dlpack.

       a  = torch.randn(M, N, device="cuda", dtype=torch.float16)
       a_ = from_dlpack(a, assumed_align=16)   # CuTe view of `a`

   What does assumed_align=16 mean?
       "Trust me — this tensor's base pointer is 16-byte aligned."
       With that promise, the compiler may emit wide vectorized loads
       (LD.128 = 16 bytes = 8x fp16 in one transaction) instead of
       conservative narrow ones. Big throughput win.

       It's safe here because torch's CUDA allocator always returns
       256-byte-aligned pointers. But it's an UNCHECKED assumption:
       lie about it (e.g. on an oddly-offset slice) and you get
       undefined behavior. Lower it to 8 or 4 if unsure.

2) THE KERNEL — what each thread does. Marked @cute.kernel.
   A thread asks "who am I?" and then does its share of work:

       tidx = thread index inside the block   (cute.arch.thread_idx())
       bidx = block  index inside the grid    (cute.arch.block_idx())
       bdim = threads per block               (cute.arch.block_dim())

3) THE HOST FUNCTION — picks the launch shape. Marked @cute.jit.
   It decides grid/block sizes and fires the kernel.

Pipeline:
       compiled = cute.compile(host_fn, *cute_tensors)
       compiled(*cute_tensors)                    # runs on GPU
       torch.testing.assert_close(c, a + b)       # check

================================================================
THE MENTAL MODEL — read this BEFORE the code
================================================================

A @cute.kernel function is NOT a function in the Python sense.
It is a description of what ONE GPU thread does.

When the kernel launches, the hardware spawns ~1,000,000 copies of
this function and runs them in parallel. Each copy executes the
body EXACTLY ONCE, with its own thread/block ids.

So when you read the kernel body, picture yourself as one
anonymous worker in a huge crowd:

    "Who am I?"             -> read thread_idx / block_idx
    "What's my unique id?"  -> i = bidx * bdim + tidx
    "Which cell is mine?"   -> row, col = i // n, i % n
    "Do my one piece."      -> gC[row, col] = gA[row, col] + gB[row, col]

This is why:

  * `i`, `row`, `col` are computed ONCE — each thread has only one
    cell to handle. There is NO Python `for` loop over elements;
    the "loop" is the hardware launching a million threads at once.

  * There is NO `return`. The result of the kernel is the SIDE
    EFFECT of writing into gC. With a million threads running,
    there is no single value to return — the memory IS the answer.

================================================================
THE KERNEL — now read the code
================================================================

Every CuTe kernel follows the same four steps:

  Step 1 — ask "who am I?"        (thread_idx, block_idx, block_dim)
  Step 2 — turn that into one     global id `i` unique across the launch
  Step 3 — map `i` to a tensor    coordinate (here: flat -> (row, col))
  Step 4 — do the work            (load, compute, store)

@cute.kernel
def add_kernel(gA, gB, gC):
    # Step 1: who am I?
    tidx, _, _ = cute.arch.thread_idx()   # 0 .. block_dim-1
    bidx, _, _ = cute.arch.block_idx()    # 0 .. grid_dim-1
    bdim, _, _ = cute.arch.block_dim()    # threads per block

    # Step 2: my unique id across the whole launch.
    i = bidx * bdim + tidx

    # Step 3: flat id -> (row, col).
    _, n = gA.shape
    row, col = i // n, i % n

    # Step 4: one element of the work.
    gC[row, col] = gA[row, col] + gB[row, col]

(With M=N=1024 and 256 threads/block, the launch covers every element
exactly once — no leftover threads, so no bounds check needed yet.)

================================================================
HOW MANY BLOCKS DO WE LAUNCH?
================================================================

One thread per element, so:

       total threads = M * N
       blocks        = (M * N) / threads_per_block

Example with M=N=1024 and 256 threads/block:
       total  = 1024 * 1024 = 1,048,576
       blocks = 1,048,576 / 256 = 4096
       launch: grid=(4096, 1, 1), block=(256, 1, 1)

================================================================
PREDICT
================================================================

Q1. With M=1024, N=1024, threads_per_block=256, how many
    THREAD BLOCKS does the launch create?
    your guess: 4096

Q2. If thread tidx=0 is in block bidx=0, what (row, col) does it
    process?
    your guess: (0, 0)

Q3. If thread tidx=5 is in block bidx=3, what global element
    index does it touch? (i.e. row*N + col)
    your guess: 518

Q4. We pick threads_per_block=256. Why is 256 (and not, say, 100)?
    your guess: to divide evenly

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
    tidx, _, _ = cute.arch.thread_idx()                                                                                                                                                                                                                                                                                                             
    bidx, _, _ = cute.arch.block_idx()                                                                                                                                                                                                                                                                                                              
    bdim, _, _ = cute.arch.block_dim()                                                                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                                                                    
    # TODO 2: compute a global thread id `i` so that every thread has a UNIQUE i.                                                                                                                                                                                                                                                                   
    #         Formula: i = (which block) * (threads per block) + (which thread in block)                                                                                                                                                                                                                                                            
    i = bidx * bdim + tidx                                                                                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                                                                    
    # TODO 3: convert flat index i to a 2D (row, col) coordinate.                                                                                                                                                                                                                                                                                   
    #         Hint: gA.shape gives (m, n). Use // and % .                                                                                                                                                                                                                                                                                           
    m, n = gA.shape                                                                                                                                                                                                                                                                                                                                 
    row = i // n                                                                                                                                                                                                                                                                                                                                    
    col = i % n                                                                                                                                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                                                                                                    
    # TODO 4: do the actual work — write A[row,col] + B[row,col] into C[row,col].                                                                                                                                                                                                                                                                   
    gC[row, col] = gA[row, col] + gB[row, col] 


@cute.jit
def add(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor) -> None:
    threads_per_block: int = 256
    m, n = mA.shape

    # TODO 5: compute number of blocks needed so that
    #         num_blocks * threads_per_block == m * n.
    num_blocks = (m * n) // threads_per_block

    add_kernel(mA, mB, mC).launch(
        grid=(num_blocks, 1, 1),
        block=(threads_per_block, 1, 1),
    )


def main() -> None:
    M: int = 1024
    N: int = 1024  # 1M elements total — small enough to be quick, big enough to be real

    a: torch.Tensor = torch.randn(M, N, device="cuda", dtype=torch.float16)
    b: torch.Tensor = torch.randn(M, N, device="cuda", dtype=torch.float16)
    c: torch.Tensor = torch.zeros(M, N, device="cuda", dtype=torch.float16)

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

"""
Puzzle #1 — Layouts: the alphabet of CuTe.

================================================================
CONCEPTS
================================================================

Core insight: GPU memory is a flat 1D array. ALWAYS.
Layouts are how we *pretend* it's 2D, 3D, hierarchical, etc.

When you allocate 8 floats, the GPU literally sees:

    addr:    0    1    2    3    4    5    6    7
    value: [ a    b    c    d    e    f    g    h ]

There are no "rows" or "columns" anywhere — those are
interpretations we layer on top with a Layout.

----------------------------------------------------------------
OFFSET — the actual address.

"Which slot in the 1D array do I read?" Offset 5 → read `f`.
That's the whole concept.

----------------------------------------------------------------
SHAPE — the lie we tell ourselves.

We *want* to think of those 8 floats as a 4x2 matrix:

         col0  col1
    row0  ?     ?
    row1  ?     ?
    row2  ?     ?
    row3  ?     ?

shape=(4, 2) says: "pretend this is a grid, 4 along dim-0,
2 along dim-1." But shape alone doesn't say WHICH memory
slot ends up in each cell. Two people could fill it differently.

----------------------------------------------------------------
STRIDE — the translation rule.

"If I take one step along dim-i, how many memory slots do I jump?"

Layout A: shape=(4,2), stride=(1,4)  -- COLUMN-MAJOR
  step along rows -> +1 in memory
  step along cols -> +4 in memory

         col0  col1
    row0   0    4
    row1   1    5
    row2   2    6     <- (2,1) = 2*1 + 1*4 = 6
    row3   3    7

  Column 0 = [0,1,2,3] — contiguous in memory.
  That's why it's "column-major": columns sit back-to-back.

Layout B: shape=(4,2), stride=(2,1)  -- ROW-MAJOR
  step along rows -> +2 in memory
  step along cols -> +1 in memory

         col0  col1
    row0   0    1
    row1   2    3
    row2   4    5     <- (2,1) = 2*2 + 1*1 = 5
    row3   6    7

  Row 0 = [0,1] contiguous; row 1 = [2,3] contiguous; etc.

----------------------------------------------------------------
THE FORMULA

For coordinate (i, j):

    offset(i, j) = i * stride[0] + j * stride[1]

Generalizes to N dimensions — just dot-product coord with stride.

----------------------------------------------------------------
WHY IT MATTERS

GPUs are fast when threads in a warp read contiguous memory
(coalesced access). Same matrix, same math — but choosing
column-major vs row-major can change speed by 5-10x. CuTe makes
the layout explicit and manipulable so you can reason about and
reshape exactly how data sits in memory before computing on it.

----------------------------------------------------------------
CHEAT SHEET

  Offset : actual address into the flat 1D buffer
  Shape  : logical grid we pretend exists, e.g. (4, 2)
  Stride : memory jump per step along each dim, e.g. (1, 4)
  Layout : (shape, stride) together — the full mapping

================================================================
PREDICT (write your guesses as comments before running):

  Q1. Layout shape=(4,2), stride=(1,4). What offset does coord (2,1) map to?
        your guess: 6

  Q2. Same layout. What offset does coord (3,0) map to?
        your guess: 3

  Q3. If we change stride to (2,1), shape stays (4,2),
      what offset does coord (2,1) map to?
        your guess: 5

  Q4. Which of the two layouts above is row-major? Which is column-major?
        your guess: (1,4) - column, (2, 1) - row

================================================================
RUN:
    python puzzles/01_layout_basics.py
"""

import cutlass
import cutlass.cute as cute


@cute.jit
def explore():
    A = cute.make_layout((4, 2), stride=(1, 4))
    B = cute.make_layout((4, 2), stride=(2, 1))

    cute.printf("Layout A: {}", A)
    cute.printf("Layout B: {}", B)

    cute.printf("A(2,1) = {}   # check Q1", A((2, 1)))
    cute.printf("A(3,0) = {}   # check Q2", A((3, 0)))
    cute.printf("B(2,1) = {}   # check Q3", B((2, 1)))

    cute.printf("Full map of A:")
    for i in cutlass.range_constexpr(4):
        for j in cutlass.range_constexpr(2):
            cute.printf("  A({},{}) -> {}", i, j, A((i, j)))


if __name__ == "__main__":
    explore()

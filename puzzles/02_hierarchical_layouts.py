"""
Puzzle #2 — Hierarchical layouts (the CuTe superpower).

================================================================
THE BIG IDEA
================================================================

In puzzle #1, shape was a flat tuple: (4, 2).
But in CuTe, each "mode" (each dim) can ITSELF be a tuple.

    shape = ((2, 2), (2, 2))

This still describes a 4x4 logical grid (2*2 = 4 along each dim),
BUT it remembers internal structure:

    dim-0 = (inner_row=2, outer_row=2)   <- 2 levels deep
    dim-1 = (inner_col=2, outer_col=2)   <- 2 levels deep

You access it with a HIERARCHICAL coordinate:

    A( ((i0, i1), (j0, j1)) )

  i0 = which row WITHIN a tile (0 or 1)
  i1 = which tile-row             (0 or 1)
  j0 = which col WITHIN a tile (0 or 1)
  j1 = which tile-col             (0 or 1)

----------------------------------------------------------------
WHY THIS IS USEFUL

A GPU GEMM does not just multiply two matrices. It tiles them:
  - thread blocks own big tiles,
  - warps own medium tiles inside that,
  - threads own small tiles inside THAT,
  - tensor cores want a specific shape inside even THAT.

A flat shape (M, N) loses all this structure.
A hierarchical shape KEEPS it — and CuTe operations
(tile, partition, compose) all work along the modes.

----------------------------------------------------------------
WORKED EXAMPLE

Take a 4x4 ROW-MAJOR matrix. Memory layout:

         col0  col1  col2  col3
    row0   0    1    2    3
    row1   4    5    6    7
    row2   8    9   10   11
    row3  12   13   14   15

Now reinterpret as 2x2 tiles of 2x2 elements:

    +----+----+        each tile is 2x2 elements
    | T00| T01|        tiles are arranged in a 2x2 grid
    +----+----+        T_ab = tile at tile-row a, tile-col b
    | T10| T11|
    +----+----+

Tile T00 contains memory offsets: 0, 1, 4, 5
Tile T01 contains memory offsets: 2, 3, 6, 7
Tile T10 contains memory offsets: 8, 9, 12, 13
Tile T11 contains memory offsets: 10, 11, 14, 15

Hierarchical layout for this:

    shape  = ((2, 2), (2, 2))    # ((inner_row, outer_row), (inner_col, outer_col))
    stride = ((4, 8), (1, 2))

Why those strides?
  - inner_row: step inside a tile, down one row -> +4 in memory (next row of full matrix)
  - outer_row: step to next TILE-row -> +8 in memory (skip 2 rows = 8 elements)
  - inner_col: step inside a tile, right one col -> +1
  - outer_col: step to next TILE-col -> +2

================================================================
PREDICT — fill in your guesses, THEN run.
================================================================

shape  = ((2, 2), (2, 2))
stride = ((4, 8), (1, 2))

Formula reminder (hierarchical):
  offset = i0*4 + i1*8 + j0*1 + j1*2

  Q1. coord ((0,0),(0,0)) -> ?     your guess: 0
        (top-left element of tile T00)

  Q2. coord ((1,0),(1,0)) -> ?     your guess: 5
        (bottom-right element of tile T00 — should be 5)

  Q3. coord ((0,1),(0,0)) -> ?     your guess: 8
        (top-left of tile T10 — start of the bottom-left tile)

  Q4. coord ((1,1),(1,1)) -> ?     your guess: 15
        (bottom-right of tile T11 — should be the very last element)

  Q5. Which (i0,i1,j0,j1) gives offset 6?     your guess: ((1, 0), (0, 1))

================================================================
"""

import cutlass
import cutlass.cute as cute


@cute.jit
def explore():
    L = cute.make_layout(((2, 2), (2, 2)), stride=((4, 8), (1, 2)))
    cute.printf("Hierarchical layout L: {}", L)

    cute.printf("Q1 ((0,0),(0,0)) -> {}", L(((0, 0), (0, 0))))
    cute.printf("Q2 ((1,0),(1,0)) -> {}", L(((1, 0), (1, 0))))
    cute.printf("Q3 ((0,1),(0,0)) -> {}", L(((0, 1), (0, 0))))
    cute.printf("Q4 ((1,1),(1,1)) -> {}", L(((1, 1), (1, 1))))

    cute.printf("Full map (tile-by-tile):")
    for i1 in cutlass.range_constexpr(2):       # outer row (which tile-row)
        for j1 in cutlass.range_constexpr(2):   # outer col (which tile-col)
            cute.printf(" tile T{}{}:", i1, j1)
            for i0 in cutlass.range_constexpr(2):       # inner row
                for j0 in cutlass.range_constexpr(2):   # inner col
                    cute.printf(
                        "   ({},{}),({},{}) -> {}",
                        i0, i1, j0, j1,
                        L(((i0, i1), (j0, j1))),
                    )


if __name__ == "__main__":
    explore()

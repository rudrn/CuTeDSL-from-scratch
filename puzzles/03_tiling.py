"""
Puzzle #3 — Tiling: stop computing strides by hand.

================================================================
THE BIG IDEA
================================================================

In puzzle #2 you wrote:

    shape  = ((2, 2), (2, 2))
    stride = ((4, 8), (1, 2))

...by carefully computing strides. Tedious. Error-prone.
Now imagine doing that for a 1024x1024 matrix tiled into
128x128 thread-block tiles, each split into 32x32 warp tiles,
each split into 16x8 tensor-core tiles. You'd lose your mind.

CuTe has an operation that does it automatically:

    cute.tiled_divide(layout, tile_shape)

Given:
  - a flat layout (the "big" matrix), e.g. shape=(4,4) row-major
  - a tile shape, e.g. (2,2)

it returns a HIERARCHICAL layout where:
  - mode-0 = "within a tile" (intra-tile coords)
  - mode-1 = "which tile"    (rest of the original space)

That's exactly the shape we hand-built last time, but derived
from "matrix shape + tile shape" — which is how you'd actually
think about a real GEMM.

----------------------------------------------------------------
WHAT WE'RE DOING IN THIS PUZZLE

Start with the natural 4x4 row-major layout:

    shape  = (4, 4)
    stride = (4, 1)        # row-major: down one row -> +4, right one col -> +1

Tile it by (2, 2). Predict:
  - what's the resulting hierarchical shape?
  - what's the resulting stride?
  - which memory offsets does tile (0,0) cover? tile (1,1)?

================================================================
PREDICT — fill in your guesses, then run.
================================================================

  Q1. After tiled_divide(shape=(4,4) stride=(4,1), tile=(2,2)),
      what's the resulting SHAPE?

      Hint: it should look like ((tile_h, tile_w), (n_tiles_h, n_tiles_w))
            = ((2, 2), (2, 2))

      your guess: ((2, 2), (2, 2))

  Q2. What's the resulting STRIDE?
      Hint: think — within a tile, stepping down a row in the
            ORIGINAL matrix is still +4. Stepping to the next
            TILE-row skips 2 original rows = +8.

      your guess: ((4, 8), (1, 2))

  Q3. Tile (0,0) is the TOP-LEFT 2x2 block of the matrix.
      Which 4 memory offsets sit inside it?
      (Look at the matrix diagram above and read off the 4 numbers
       in the top-left 2x2 corner.)

      your guess: {0, 1, 4, 5}

  Q4. Tile (1,1) is the BOTTOM-RIGHT 2x2 block of the matrix.
      Which 4 memory offsets sit inside it?
      (Same idea: look at the bottom-right 2x2 corner of:
              col0  col1  col2  col3
         row0   0    1    2    3
         row1   4    5    6    7
         row2   8    9   10   11
         row3  12   13   14   15
       and write the 4 numbers as a set.)

      your guess: {10, 11, 14, 15}

================================================================
"""

import cutlass
import cutlass.cute as cute


@cute.jit
def explore():
    # Start with a plain 4x4 row-major layout. No hierarchy yet.
    flat = cute.make_layout((4, 4), stride=(4, 1))
    cute.printf("flat layout:        {}", flat)

    # Ask CuTe: "tile this by (2,2)". It hands us back a hierarchical layout.
    tiled = cute.tiled_divide(flat, (2, 2))
    cute.printf("tiled layout:       {}", tiled)

    # Now mode-0 is "inside a tile", mode-1 is "which tile".
    # Walk it: outer = which tile, inner = positions within that tile.
    cute.printf("Walk tile-by-tile:")
    for ti in cutlass.range_constexpr(2):       # which tile-row
        for tj in cutlass.range_constexpr(2):   # which tile-col
            cute.printf(" tile ({},{}):", ti, tj)
            for ii in cutlass.range_constexpr(2):       # within-tile row
                for jj in cutlass.range_constexpr(2):   # within-tile col
                    # Coordinate is ((within_row, within_col), (tile_row, tile_col))
                    cute.printf(
                        "   intra=({},{}) -> offset {}",
                        ii, jj,
                        tiled(((ii, jj), ti, tj)),
                    )


################################################################
# YOUR TURN — write code below. Each TASK is a small change.
# After each one, run the file and check the output.
################################################################


# ----------------------------------------------------------------
# TASK 1 — Bigger matrix, same tile size.
#
# Goal: tile an 8x8 row-major matrix by (2,2) tiles.
# Expected: the tile layout should give 4x4 = 16 tiles of 4 elements.
#
# Fill in the blanks (the `???`), then call task1() from main.
# ----------------------------------------------------------------
@cute.jit
def task1():
    # An 8x8 row-major matrix has stride (?, 1) — what's the row stride?
    flat = cute.make_layout((8, 8), stride=(8, 1))
    tiled = cute.tiled_divide(flat, (2, 2))
    cute.printf("TASK 1 tiled: {}", tiled)
    # Print just tile (2, 3) — the tile in tile-row 2, tile-col 3.
    cute.printf("Tile (2,3):")
    for ii in cutlass.range_constexpr(2):
        for jj in cutlass.range_constexpr(2):
            cute.printf("  intra=({},{}) -> {}", ii, jj, tiled(((ii, jj), 2, 3)))
    # PREDICT BEFORE RUNNING: which 4 offsets should tile (2,3) cover?
    # your guess: {38, 39, 46, 47}


# ----------------------------------------------------------------
# TASK 2 — Bigger tile.
#
# Same 8x8 row-major matrix, but tile by (4,4) — bigger tiles.
# How many tiles? How big is each tile? Print all of tile (1,1).
# ----------------------------------------------------------------
@cute.jit
def task2():
    flat = cute.make_layout((8, 8), stride=(8, 1))
    tiled = cute.tiled_divide(flat, (4, 4))  # the tile shape
    cute.printf("TASK 2 tiled: {}", tiled)
    cute.printf("Tile (1,1):")
    for ii in cutlass.range_constexpr(4):
        for jj in cutlass.range_constexpr(4):
            cute.printf("  intra=({},{}) -> {}", ii, jj, tiled(((ii, jj), 1, 1)))


# ----------------------------------------------------------------
# TASK 3 — Non-square: rectangular tile.
#
# 8x8 row-major matrix, tile by (2, 4) — short&wide tiles.
# Print tile (3, 1) — the bottom-left-ish region.
# Predict the 8 offsets first.
# ----------------------------------------------------------------
@cute.jit
def task3():
    flat = cute.make_layout((8, 8), stride=(8, 1))
    tiled = cute.tiled_divide(flat, (2, 4))
    cute.printf("TASK 3 tiled: {}", tiled)
    cute.printf("Tile (3,1):")
    for ii in cutlass.range_constexpr(2):
        for jj in cutlass.range_constexpr(4):
            cute.printf("  intra=({},{}) -> {}", ii, jj, tiled(((ii, jj), 3, 1)))
    # PREDICT: tile (3,1) should be rows {?,?} cols {?,?,?,?} of the original.
    # your guess: rows {6, 7} cols {4, 5, 6, 7}


# ----------------------------------------------------------------
# TASK 4 — COLUMN-major matrix this time.
#
# Same 8x8 elements, but stored column-major: stride=(1, 8).
# Tile by (2, 2). Print tile (0, 0). The OFFSETS will look very
# different from the row-major version above. Predict first.
# ----------------------------------------------------------------
@cute.jit
def task4():
    flat = cute.make_layout((8, 8), stride=(1, 8))  # column-major: which stride?
    tiled = cute.tiled_divide(flat, (2, 2))
    cute.printf("TASK 4 tiled: {}", tiled)
    cute.printf("Tile (0,0):")
    for ii in cutlass.range_constexpr(2):
        for jj in cutlass.range_constexpr(2):
            cute.printf("  intra=({},{}) -> {}", ii, jj, tiled(((ii, jj), 0, 0)))
    # PREDICT: in column-major 8x8, the top-left 2x2 block has offsets {?,?,?,?}.
    # your guess: {0, 1, 8, 9}


# ----------------------------------------------------------------
# TASK 5 (harder) — Where does a given offset live?
#
# In an 8x8 row-major matrix tiled by (2,2):
# The element at original (row=5, col=3) — which TILE is it in,
# and what's its INTRA-tile coord?
#
# Hint: tile_row = row // 2, tile_col = col // 2,
#       intra_row = row % 2, intra_col = col % 2.
# Verify by computing the offset two ways:
#   (a) directly from the flat layout: flat((5, 3))
#   (b) via the tiled layout: tiled(((??, ??), ??, ??))
# Both must match.
# ----------------------------------------------------------------
@cute.jit
def task5():
    flat = cute.make_layout((8, 8), stride=(8, 1))
    tiled = cute.tiled_divide(flat, (2, 2))
    direct = flat((5, 3))
    via_tile = tiled(((1, 1), 2, 1))
    cute.printf("direct={}  via_tile={}  (must match)", direct, via_tile)


if __name__ == "__main__":
    explore()
    # Uncomment as you complete each task:
    task1()
    task2()
    task3()
    task4()
    task5()

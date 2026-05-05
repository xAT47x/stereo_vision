# Assignment 3 - Part 1.2 and Bonus

## Idea

Part 1.2 computes stereo disparity one row at a time. For every row, the code
tries to align pixels from the left image with pixels from the right image.

Each step has three choices:

1. Match a left pixel with a right pixel.
2. Skip a left pixel because it is occluded.
3. Skip a right pixel because it is occluded.

The dynamic programming table stores the cheapest cost found so far. After the
table is complete, backtracking gives the best alignment path.

## Pixel Matching Cost

```python
diff = float(left_pixel) - float(right_pixel)
cost = (diff ** 2) / (2 * (sigma ** 2))
return cost
```

This is the assignment equation. If two pixels have similar gray values, the
difference is small, so the cost is small. The assignment uses `sigma = 2`.

## Dynamic Programming Table

```python
D = np.full((n + 1, n + 1), np.inf, dtype=np.float32)
moves = np.empty((n + 1, n + 1), dtype=object)
D[0, 0] = 0
```

`D[i, j]` means: what is the cheapest cost for aligning the first `i`
pixels from the left row with the first `j` pixels from the right row?

`moves` remembers which choice was best, so we can trace the answer later.

## Three Choices at Each Cell

```python
match_cost = D[i - 1, j - 1] + get_pixel_cost(...)
skip_left_cost = D[i - 1, j] + c0
skip_right_cost = D[i, j - 1] + c0
```

`match` is a diagonal step. It means the two pixels correspond to each other.

`skip_left` is a vertical step. It means one left pixel is occluded.

`skip_right` is a horizontal step. It means one right pixel is occluded.

The assignment uses `c0 = 1`.

## Backtracking

```python
while i > 0 or j > 0:
    move = moves[i, j]
```

Backtracking starts at the bottom-right corner of the table and walks backwards.
This recovers the actual matches and skips that produced the minimum cost.

When a match is found:

```python
disparity = abs(i - j)
left_disparity[i - 1] = disparity
right_disparity[j - 1] = disparity
```

Disparity is the horizontal shift between the matched left and right pixel
positions. Skipped pixels get disparity `0`.

## Bonus Plot

```python
plt.plot(right_positions, left_positions)
```

The x-axis is the right scanline position, and the y-axis is the left scanline
position.

Diagonal lines mean matched pixels, horizontal lines mean right pixels were
skipped, and vertical lines mean left pixels were skipped.

## How To Run

From the `stereo_vision` folder:

```bash
python dynamic_programming_demo.py
```

The script saves results in:

```text
outputs/
```

By default it runs only image pair 1. To run all image pairs, open
`dynamic_programming_demo.py` and change:

```python
pairs = [1]
```

to:

```python
pairs = [1, 2, 3]
```

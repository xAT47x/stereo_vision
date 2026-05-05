import numpy as np


def get_pixel_cost(left_pixel, right_pixel, sigma=2):
    diff = float(left_pixel) - float(right_pixel)
    cost = (diff ** 2) /  (sigma ** 2)
    return cost


def match_one_row(left_row, right_row, sigma=2, c0=1):
    
    left_row = left_row.astype(np.float32)
    right_row = right_row.astype(np.float32)

    n = len(left_row)
    # I used n + 1 because row 0 and column 0 mean "no pixels yet".
    D = np.full((n + 1, n + 1), np.inf, dtype=np.float32)

    # This table remembers where each value in D came from.
    # It helps us during backtracking.
    moves = np.empty((n + 1, n + 1), dtype=object)

    D[0, 0] = 0
    moves[0, 0] = "start"

    # First column: the right row has no pixels yet, so we skip left pixels.
    for i in range(1, n + 1):
        D[i, 0] = D[i - 1, 0] + c0
        moves[i, 0] = "skip_left"

    # First row: the left row has no pixels yet, so we skip right pixels.
    for j in range(1, n + 1):
        D[0, j] = D[0, j - 1] + c0
        moves[0, j] = "skip_right"

    # Fill the dynamic programming table.
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            match_cost = D[i - 1, j - 1] + get_pixel_cost(
                left_row[i - 1],
                right_row[j - 1],
                sigma,
            )

            skip_left_cost = D[i - 1, j] + c0
            skip_right_cost = D[i, j - 1] + c0

            # Choose the cheapest of the three options.
            # If two options are equal, I prefer matching the pixels.
            best_cost = min(match_cost, skip_left_cost, skip_right_cost)
            D[i, j] = best_cost

            if best_cost == match_cost:
                moves[i, j] = "match"
            elif best_cost == skip_left_cost:
                moves[i, j] = "skip_left"
            else:
                moves[i, j] = "skip_right"

    left_disparity = np.zeros(n, dtype=np.float32)
    right_disparity = np.zeros(n, dtype=np.float32)

    # The path will be used in the bonus plot.
    path = []

    # Start from the last cell and move back to the first cell.
    i = n
    j = n
    path.append((i, j))

    while i > 0 or j > 0:
        move = moves[i, j]

        if move == "match":
            # i and j are positions in the DP table.
            # The shift between them is the disparity.
            disparity = abs(i - j)
            left_disparity[i - 1] = disparity
            right_disparity[j - 1] = disparity
            i = i - 1
            j = j - 1

        elif move == "skip_left":
            # A skipped pixel gets disparity 0.
            left_disparity[i - 1] = 0
            i = i - 1

        else:
            # Here move is "skip_right".
            right_disparity[j - 1] = 0
            j = j - 1

        path.append((i, j))

    # The path was collected backwards, so I reverse it for plotting.
    path.reverse()

    return left_disparity, right_disparity, D, path


def dynamic_programming_disparity(left_image, right_image, sigma=2, c0=1):
    """
    Compute the disparity map for the whole image.
    take one row from the left image and one row from the right image,
    solve them using match_one_row, then go to the next row.
    """
    if left_image.shape != right_image.shape:
        raise ValueError("The two images must have the same size.")

    height, width = left_image.shape

    left_map = np.zeros((height, width), dtype=np.float32)
    right_map = np.zeros((height, width), dtype=np.float32)

    for row in range(height):
        left_disparity, right_disparity, D, path = match_one_row(
            left_image[row],
            right_image[row],
            sigma,
            c0,
        )

        left_map[row] = left_disparity
        right_map[row] = right_disparity

    return left_map, right_map

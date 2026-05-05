import os
import cv2
from utils.dynamic_programming import dynamic_programming_disparity, match_one_row
from utils.plots import plot_alignment_path

def read_gray_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError("Image was not found: " + path)
    return image


def save_disparity_image(disparity, path):
    image_to_save = cv2.normalize(
        disparity,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
        cv2.CV_8U,
    )
    cv2.imwrite(path, image_to_save)

def run_pair(pair_number):
    folder = os.path.dirname(__file__)
    images_folder = os.path.join(folder, "imgs")
    output_folder = os.path.join(folder, "outputs")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    left_path = os.path.join(images_folder, "l" + str(pair_number) + ".png")
    right_path = os.path.join(images_folder, "r" + str(pair_number) + ".png")

    left_image = read_gray_image(left_path)
    right_image = read_gray_image(right_path)

    sigma = 2
    c0 = 1

    print("Running dynamic programming for pair", pair_number)

    left_disparity, right_disparity = dynamic_programming_disparity(
        left_image,
        right_image,
        sigma,
        c0,
    )

    left_output = os.path.join(
        output_folder,
        "pair" + str(pair_number) + "_dp_left_disparity.png",
    )
    right_output = os.path.join(
        output_folder,
        "pair" + str(pair_number) + "_dp_right_disparity.png",
    )

    save_disparity_image(left_disparity, left_output)
    save_disparity_image(right_disparity, right_output)

    # Bonus part:
    # I chose the middle row because it usually passes through useful details.
    row_number = left_image.shape[0] // 2

    left_row = left_image[row_number]
    right_row = right_image[row_number]

    row_left_disp, row_right_disp, D, path = match_one_row(
        left_row,
        right_row,
        sigma,
        c0,
    )

    bonus_output = os.path.join(
        output_folder,
        "pair" + str(pair_number) + "_bonus_alignment_row_" + str(row_number) + ".png",
    )

    plot_alignment_path(
        path,
        save_path=bonus_output,
        pair_name="pair " + str(pair_number),
        row_index=row_number,
    )

    print("Saved:", left_output)
    print("Saved:", right_output)
    print("Saved:", bonus_output)


if __name__ == "__main__":
    pairs = [1,2,3]
    for pair_number in pairs:
        run_pair(pair_number)

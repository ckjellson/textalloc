import numpy as np
from typing import Tuple
from textalloc.candidates import generate_candidates
from textalloc.overlap_functions import (
    non_overlapping_with_points,
    non_overlapping_with_lines,
    non_overlapping_with_rectangles,
    inside_plot,
)
from tqdm import tqdm


def get_non_overlapping_patches(
    original_patches: np.ndarray,
    xlims: Tuple[float, float],
    ylims: Tuple[float, float],
    distance_margin_fraction: float,
    verbose: bool,
    scatter_xy: np.ndarray = None,
    lines_xyxy: np.ndarray = None,
):
    # """
    # Parameters
    # ----------
    # original_patches : list
    #     List of tuples containing width, height and term of each original text-
    #     box (xmin,ymin,w,h,s) for all N original patches
    # pointarr : np.ndarray
    #     Array of shape (N,2) containing coordinates for all scatter-points
    # xlims : tuple
    #     (xmin, xmax) of plot
    # ylims : tuple
    #     (ymin, ymax) of plot
    # distance_margin_fraction : float
    #     Fraction of the 2d space to use as margins for text bboxes

    # Returns
    # -------
    # list
    # List of tuples containing x, y-coordinates and term of each non-overlapping text-
    # box (xmin,ymin,w,h,s,ind) considering both other patches and the scatterplot-points

    # """
    xmin_bound, xmax_bound = xlims
    ymin_bound, ymax_bound = ylims

    xfrac = (xmax_bound - xmin_bound) * distance_margin_fraction
    yfrac = (ymax_bound - ymin_bound) * distance_margin_fraction

    rectangle_arr = np.zeros((0, 4))
    non_overlapping_patches = []

    # Iterate original patches and find ones that do not overlap by creating multiple candidates
    non_overlapping_patches = []
    for i, patch in tqdm(enumerate(original_patches), disable=not verbose):
        x_original, y_original, w, h, s = patch
        candidates = generate_candidates(
            w, h, x_original, y_original, xfrac / 2, yfrac / 2
        )

        # Check for overlapping
        if scatter_xy is None:
            non_op = np.zeros((candidates.shape[0],)) == 0
        else:
            non_op = non_overlapping_with_points(
                scatter_xy, candidates, xfrac / 2, yfrac / 2
            )
        if lines_xyxy is None:
            non_ol = np.zeros((candidates.shape[0],)) == 0
        else:
            non_ol = non_overlapping_with_lines(
                lines_xyxy, candidates, xfrac / 2, yfrac / 2
            )
        if rectangle_arr.shape[0] == 0:
            non_orec = np.zeros((candidates.shape[0],)) == 0
        else:
            non_orec = non_overlapping_with_rectangles(
                rectangle_arr, candidates, xfrac / 2, yfrac / 2
            )
        inside = inside_plot(xmin_bound, ymin_bound, xmax_bound, ymax_bound, candidates)

        # Validate
        ok_candidates = np.where(
            np.bitwise_and(
                non_ol, np.bitwise_and(non_op, np.bitwise_and(non_orec, inside))
            )
        )[0]
        if len(ok_candidates) > 0:
            best_candidate = candidates[ok_candidates[0], :]
            rectangle_arr = np.vstack(
                [
                    rectangle_arr,
                    np.array(
                        [
                            best_candidate[0],
                            best_candidate[1],
                            best_candidate[0] + w,
                            best_candidate[1] + h,
                        ]
                    ),
                ]
            )
            non_overlapping_patches.append(
                (best_candidate[0], best_candidate[1], w, h, s, i)
            )
    return non_overlapping_patches

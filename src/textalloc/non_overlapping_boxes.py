import numpy as np
from typing import Tuple, List
from textalloc.candidates import generate_candidates
from textalloc.overlap_functions import (
    non_overlapping_with_points,
    non_overlapping_with_lines,
    non_overlapping_with_boxes,
    inside_plot,
)
from tqdm import tqdm


def get_non_overlapping_boxes(
    original_boxes: np.ndarray,
    xlims: Tuple[float, float],
    ylims: Tuple[float, float],
    distance_margin_fraction: float,
    verbose: bool,
    nbr_candidates: int,
    scatter_xy: np.ndarray = None,
    lines_xyxy: np.ndarray = None,
) -> Tuple[List[Tuple[float, float, float, float, str, int]], List[int]]:
    """Finds boxes that do not have an overlap with any other objects.

    Args:
        original_boxes (np.ndarray): original boxes containing texts.
        xlims (Tuple[float, float]): x-limits of plot gotten from ax.get_ylim()
        ylims (Tuple[float, float]): y-limits of plot gotten from ax.get_ylim()
        distance_margin_fraction (float): parameter for margins between objects. Increase for larger margins to points and lines.
        verbose (bool): prints progress using tqdm.
        nbr_candidates (int): Sets the number of candidates used.
        scatter_xy (np.ndarray, optional): 2d array of scattered points in plot. Defaults to None.
        lines_xyxy (np.ndarray, optional): 2d array of line segments in plot.. Defaults to None.

    Returns:
        Tuple[List[Tuple[float, float, float, float, str, int]], List[int]]: data of non-overlapping boxes and indices of overlapping boxes.
    """
    xmin_bound, xmax_bound = xlims
    ymin_bound, ymax_bound = ylims

    xfrac = (xmax_bound - xmin_bound) * distance_margin_fraction
    yfrac = (ymax_bound - ymin_bound) * distance_margin_fraction

    box_arr = np.zeros((0, 4))
    non_overlapping_boxes = []

    # Iterate original boxes and find ones that do not overlap by creating multiple candidates
    non_overlapping_boxes = []
    overlapping_boxes_inds = []
    for i, box in tqdm(enumerate(original_boxes), disable=not verbose):
        x_original, y_original, w, h, s = box
        candidates = generate_candidates(
            w,
            h,
            x_original,
            y_original,
            xfrac / 2,
            yfrac / 2,
            nbr_candidates=nbr_candidates,
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
        if box_arr.shape[0] == 0:
            non_orec = np.zeros((candidates.shape[0],)) == 0
        else:
            non_orec = non_overlapping_with_boxes(
                box_arr, candidates, xfrac / 2, yfrac / 2
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
            box_arr = np.vstack(
                [
                    box_arr,
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
            non_overlapping_boxes.append(
                (best_candidate[0], best_candidate[1], w, h, s, i)
            )
        else:
            overlapping_boxes_inds.append(i)
    return non_overlapping_boxes, overlapping_boxes_inds

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
    original_boxes: list,
    xlims: Tuple[float, float],
    ylims: Tuple[float, float],
    margin: float,
    min_distance: float,
    max_distance: float,
    verbose: bool,
    nbr_candidates: int,
    draw_all: bool,
    scatter_xy: np.ndarray,
    lines_xyxy: np.ndarray,
    scatter_sizes: np.ndarray,
    text_scatter_sizes: np.ndarray,
) -> Tuple[List[Tuple[float, float, float, float, str, int]], List[int]]:
    """Finds boxes that do not have an overlap with any other objects.

    Args:
        original_boxes (np.ndarray): original boxes containing texts.
        xlims (Tuple[float, float]): x-limits of plot gotten from ax.get_ylim()
        ylims (Tuple[float, float]): y-limits of plot gotten from ax.get_ylim()
        margin (float): parameter for margins between objects. Increase for larger margins to points and lines.
        min_distance (float): parameter for max distance between text and origin.
        max_distance (float): parameter for max distance between text and origin.
        verbose (bool): prints progress using tqdm.
        nbr_candidates (int): Sets the number of candidates used.
        draw_all (bool): Draws all texts after allocating as many as possible despit overlap.
        scatter_xy (np.ndarray, optional): 2d array of scattered points in plot.
        lines_xyxy (np.ndarray, optional): 2d array of line segments in plot.
        scatter_sizes (array-like): array of object sizes with centers in scatter_xy.
        text_scatter_sizes (array-like): array of object sizes with centers in text objects.

    Returns:
        Tuple[List[Tuple[float, float, float, float, str, int]], List[int]]: data of non-overlapping boxes and indices of overlapping boxes.
    """
    xmin_bound, xmax_bound = xlims
    ymin_bound, ymax_bound = ylims

    xmargin = (xmax_bound - xmin_bound) * margin
    ymargin = (ymax_bound - ymin_bound) * margin
    xmindistance = (xmax_bound - xmin_bound) * min_distance
    ymindistance = (ymax_bound - ymin_bound) * min_distance
    xmaxdistance = (xmax_bound - xmin_bound) * max_distance
    ymaxdistance = (ymax_bound - ymin_bound) * max_distance

    box_arr = np.zeros((0, 4))
    non_overlapping_boxes = []
    has_text_scatter_sizes = text_scatter_sizes is not None
    if has_text_scatter_sizes:
        assert len(text_scatter_sizes) == len(original_boxes)
    if scatter_sizes is not None and scatter_xy is not None:
        assert len(scatter_sizes) == scatter_xy.shape[0]

    # Iterate original boxes and find ones that do not overlap by creating multiple candidates
    non_overlapping_boxes = []
    overlapping_boxes_inds = []
    for i, box in tqdm(enumerate(original_boxes), disable=not verbose):
        x_original, y_original, w, h, s = box
        text_scatter_size = 0
        if has_text_scatter_sizes:
            text_scatter_size = text_scatter_sizes[i]
        candidates = generate_candidates(
            w,
            h,
            x_original,
            y_original,
            xmindistance,
            ymindistance,
            xmaxdistance,
            ymaxdistance,
            nbr_candidates,
            scatter_size=text_scatter_size,
        )

        # Check for overlapping
        if scatter_xy is None:
            non_op = np.zeros((candidates.shape[0],)) == 0
        else:
            non_op = non_overlapping_with_points(
                scatter_xy,
                candidates,
                xmargin,
                ymargin,
                scatter_sizes,
            )
        if lines_xyxy is None:
            non_ol = np.zeros((candidates.shape[0],)) == 0
        else:
            non_ol = non_overlapping_with_lines(
                lines_xyxy, candidates, xmargin, ymargin
            )
        if box_arr.shape[0] == 0:
            non_orec = np.zeros((candidates.shape[0],)) == 0
        else:
            non_orec = non_overlapping_with_boxes(box_arr, candidates, xmargin, ymargin)
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
            if draw_all:
                ok_candidates = np.where(np.bitwise_and(non_orec, inside))[0]
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
            else:
                overlapping_boxes_inds.append(i)
    return non_overlapping_boxes, overlapping_boxes_inds

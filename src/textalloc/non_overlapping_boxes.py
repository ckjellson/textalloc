import numpy as np
from typing import Tuple, List
from textalloc.candidates import generate_candidates
from textalloc.overlap_functions import (
    non_overlapping_with_points,
    non_overlapping_with_lines,
    non_overlapping_with_boxes,
    inside_plot,
)

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterator, *args, **kwargs):
        return iterator


def get_non_overlapping_boxes(
    original_boxes: list,
    xlims: Tuple[float, float],
    ylims: Tuple[float, float],
    aspect_ratio: float,
    margin: float,
    min_distance: float,
    max_distance: float,
    verbose: bool,
    nbr_candidates: int,
    draw_all: bool,
    scatter_xy: np.ndarray,
    lines_xyxy: np.ndarray,
    scatter_sizes: np.ndarray,
    scatter_plot_bbs: np.ndarray,
    text_scatter_sizes: np.ndarray,
    direction: str,
    draw_lines: bool,
    avoid_label_lines_overlap: bool,
) -> Tuple[List[Tuple[float, float, float, float, str, int]], List[int]]:
    """Finds boxes that do not have an overlap with any other objects.

    Args:
        original_boxes (np.ndarray): original boxes containing texts.
        xlims (Tuple[float, float]): x-limits of plot gotten from ax.get_ylim()
        ylims (Tuple[float, float]): y-limits of plot gotten from ax.get_ylim()
        aspect_ratio (float): aspect ratio of the fig.
        margin (float): parameter for margins between objects. Increase for larger margins to points and lines.
        min_distance (float): parameter for max distance between text and origin.
        max_distance (float): parameter for max distance between text and origin.
        verbose (bool): prints progress using tqdm.
        nbr_candidates (int): Sets the number of candidates used.
        draw_all (bool): Draws all texts after allocating as many as possible despit overlap.
        scatter_xy (np.ndarray): 2d array of scattered points in plot.
        lines_xyxy (np.ndarray): 2d array of line segments in plot.
        scatter_sizes (array-like): array of object sizes with centers in scatter_xy.
        scatter_plot_bbs (np.ndarray): boxes extracted from scatter plot.
        text_scatter_sizes (array-like): array of object sizes with centers in text objects.
        direction (str): set preferred direction of the boxes.
        draw_lines (bool): draws lines from original points to textboxes.
        avoid_label_lines_overlap (bool): If True, avoids overlap with lines drawn between text labels and locations.

    Returns:
        Tuple[List[Tuple[float, float, float, float, str, int]], List[int]]: data of non-overlapping boxes and indices of overlapping boxes.
    """
    xmin_bound, xmax_bound = xlims
    ymin_bound, ymax_bound = ylims
    xdiff = xmax_bound - xmin_bound
    ydiff = ymax_bound - ymin_bound

    xmargin = xdiff * margin
    ymargin = ydiff * margin * aspect_ratio
    xmindistance = xdiff * min_distance
    ymindistance = ydiff * min_distance * aspect_ratio
    xmaxdistance = xdiff * max_distance
    ymaxdistance = ydiff * max_distance * aspect_ratio

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
            text_scatter_size,
            direction,
        )
        # If overlap with drawn lines should be avoided, create cand_lines.
        cand_lines = None
        if avoid_label_lines_overlap:
            for i_ in range(candidates.shape[0]):
                x_near, y_near = find_nearest_point_on_box(
                    candidates[i_, 0],
                    candidates[i_, 1],
                    w,
                    h,
                    x_original,
                    y_original,
                )
                if x_near is None:
                    x_near = x_original
                    y_near = y_original
                new_line = np.array([[x_near, y_near, x_original, y_original]])
                if cand_lines is None:
                    cand_lines = new_line
                else:
                    cand_lines = np.vstack([cand_lines, new_line])

        # Check for overlapping
        if scatter_xy is None and scatter_plot_bbs is None:
            non_op = np.zeros((candidates.shape[0],)) == 0
        elif scatter_plot_bbs is None:
            non_op = non_overlapping_with_points(
                scatter_xy,
                candidates,
                xmargin,
                ymargin,
                scatter_sizes,
            )
        else:
            non_op = non_overlapping_with_boxes(
                scatter_plot_bbs, candidates, xmargin, ymargin
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
        if cand_lines is None or box_arr.shape[0] == 0:
            non_oll = np.zeros((candidates.shape[0],)) == 0
        else:
            non_oll = non_overlapping_with_lines(
                cand_lines, box_arr, xmargin, ymargin, axis=0
            )

        # Validate
        ok_candidates = np.where(
            np.bitwise_and(
                non_ol,
                np.bitwise_and(
                    non_op,
                    np.bitwise_and(non_orec, np.bitwise_and(inside, non_oll)),
                ),
            )
        )[0]
        best_candidate = None
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
        if draw_lines and avoid_label_lines_overlap and best_candidate is not None:
            x_near, y_near = find_nearest_point_on_box(
                best_candidate[0], best_candidate[1], w, h, x_original, y_original
            )
            if x_near is not None:
                new_line = np.array([[x_near, y_near, x_original, y_original]])
                if lines_xyxy is None:
                    lines_xyxy = new_line
                else:
                    lines_xyxy = np.vstack([lines_xyxy, new_line])
    return non_overlapping_boxes, overlapping_boxes_inds


def find_nearest_point_on_box(
    xmin: float, ymin: float, w: float, h: float, x: float, y: float
) -> Tuple[float, float]:
    """Finds nearest point on box from point.
    Returns None,None if point inside box

    Args:
        xmin (float): xmin of box
        ymin (float): ymin of box
        w (float): width of box
        h (float): height of box
        x (float): x-coordinate of point
        y (float): y-coordinate of point

    Returns:
        Tuple[float, float]: x,y coordinate of nearest point
    """
    xmax = xmin + w
    ymax = ymin + h
    if x < xmin:
        if y < ymin:
            return xmin, ymin
        elif y > ymax:
            return xmin, ymax
        else:
            return xmin, y
    elif x > xmax:
        if y < ymin:
            return xmax, ymin
        elif y > ymax:
            return xmax, ymax
        else:
            return xmax, y
    else:
        if y < ymin:
            return x, ymin
        elif y > ymax:
            return x, ymax
    return None, None

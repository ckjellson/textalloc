# MIT License

# Copyright (c) 2022 Christoffer Kjellson

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from typing import Tuple, List, Union, Callable, Dict
from textalloc.candidates import generate_candidates
from textalloc.overlap_functions import (
    non_overlapping_with_points,
    non_overlapping_with_lines,
    non_overlapping_with_boxes,
    inside_plot,
    line_intersect,
)

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterator, *args, **kwargs):
        return iterator


def priority_strategy_largest(w: float, h: float) -> float:
    """Sort function for greedy text allocation."""
    return max(w, h)


PRIORITY_STRATEGIES: Dict[str, Callable[[float, float], float]] = {
    "largest": priority_strategy_largest,
}


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
    avoid_crossing_label_lines: bool,
    priority_strategy: Union[int, str, Callable[[float, float], float]],
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
        avoid_crossing_label_lines (bool): If True, avoids crossing label lines.
        priority_strategy (Union[int, str, Callable[[float, float], float]], optional): Set priority strategy for greedy text allocation
            (None / random seed / strategy name among ["largest"] / priority score of a box (width, height), the larger the better).

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
    has_text_scatter_sizes = text_scatter_sizes is not None
    if has_text_scatter_sizes:
        assert len(text_scatter_sizes) == len(original_boxes)
    if scatter_sizes is not None and scatter_xy is not None:
        assert len(scatter_sizes) == scatter_xy.shape[0]

    if priority_strategy is None:
        argsort_priority = np.arange(len(original_boxes))
    elif isinstance(priority_strategy, int):
        argsort_priority = np.random.RandomState(priority_strategy).permutation(
            len(original_boxes)
        )
    else:
        if isinstance(priority_strategy, str):
            assert (
                priority_strategy in PRIORITY_STRATEGIES
            ), f"Unknown priority strategy: {priority_strategy}. Expected one of {list(PRIORITY_STRATEGIES.keys())}"
            priority_strategy = PRIORITY_STRATEGIES[priority_strategy]
        assert isinstance(
            priority_strategy, Callable
        ), "Priority strategy must be callable"
        argsort_priority = np.argsort(
            [priority_strategy(w, h) for (_, _, w, h, _) in original_boxes]
        )[::-1]

    # Iterate original boxes and find ones that do not overlap by creating multiple candidates
    non_overlapping_boxes = []
    overlapping_boxes_inds = []
    previous_lines_xyxy = None
    for i in tqdm(argsort_priority, disable=not verbose):
        x_original, y_original, w, h, s = original_boxes[i]
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
        if avoid_label_lines_overlap or avoid_crossing_label_lines:
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

        if not avoid_label_lines_overlap or box_arr.shape[0] == 0:
            non_oll = np.zeros((candidates.shape[0],)) == 0
        else:
            assert cand_lines is not None
            non_oll = non_overlapping_with_lines(
                cand_lines, box_arr, xmargin, ymargin, axis=0
            )

        if not avoid_crossing_label_lines or box_arr.shape[0] == 0:
            non_cl = np.zeros((candidates.shape[0],)) == 0
        else:
            assert cand_lines is not None
            assert previous_lines_xyxy is not None
            non_cl = np.logical_not(
                np.any(line_intersect(cand_lines, previous_lines_xyxy), axis=-1)
            )

        # Validate
        ok_candidates = np.where(
            np.bitwise_and.reduce((non_ol, non_op, non_orec, inside, non_oll, non_cl))
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
        if (
            draw_lines
            and (avoid_label_lines_overlap or avoid_crossing_label_lines)
            and best_candidate is not None
        ):
            x_near, y_near = find_nearest_point_on_box(
                best_candidate[0], best_candidate[1], w, h, x_original, y_original
            )
            if x_near is not None:
                new_line = np.array([[x_near, y_near, x_original, y_original]])
                if avoid_label_lines_overlap:
                    if lines_xyxy is None:
                        lines_xyxy = new_line
                    else:
                        lines_xyxy = np.vstack([lines_xyxy, new_line])

                if avoid_crossing_label_lines:
                    if previous_lines_xyxy is None:
                        previous_lines_xyxy = new_line
                    else:
                        previous_lines_xyxy = np.vstack([previous_lines_xyxy, new_line])

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

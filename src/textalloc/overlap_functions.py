import numpy as np


def non_overlapping_with_points(
    scatter_xy: np.ndarray,
    candidates: np.ndarray,
    xmargin: float,
    ymargin: float,
    scatter_sizes,
) -> np.ndarray:
    """Finds candidates not overlapping with points.

    Args:
        scatter_xy (np.ndarray): Array of shape (N,2) containing coordinates for all scatter-points
        candidates (np.ndarray): Array of shape (K,4) with K candidate boxes
        xmargin (float): fraction of the x-dimension to use as margins for text boxes
        ymargin (float): fraction of the y-dimension to use as margins for text boxes

    Returns:
        np.ndarray: Boolean array of shape (K,) with True for non-overlapping candidates with points
    """
    if scatter_sizes is None:
        return np.invert(
            np.bitwise_or.reduce(
                np.bitwise_and(
                    candidates[:, 0][:, None] - xmargin < scatter_xy[:, 0],
                    np.bitwise_and(
                        candidates[:, 2][:, None] + xmargin > scatter_xy[:, 0],
                        np.bitwise_and(
                            candidates[:, 1][:, None] - ymargin < scatter_xy[:, 1],
                            candidates[:, 3][:, None] + ymargin > scatter_xy[:, 1],
                        ),
                    ),
                ),
                axis=1,
            )
        )
    else:
        return np.invert(
            np.bitwise_or.reduce(
                np.bitwise_and(
                    candidates[:, 0][:, None] - (xmargin + scatter_sizes)
                    < scatter_xy[:, 0],
                    np.bitwise_and(
                        candidates[:, 2][:, None] + (xmargin + scatter_sizes)
                        > scatter_xy[:, 0],
                        np.bitwise_and(
                            candidates[:, 1][:, None] - (ymargin + scatter_sizes)
                            < scatter_xy[:, 1],
                            candidates[:, 3][:, None] + (ymargin + scatter_sizes)
                            > scatter_xy[:, 1],
                        ),
                    ),
                ),
                axis=1,
            )
        )


def non_overlapping_with_lines(
    lines_xyxy: np.ndarray,
    candidates: np.ndarray,
    xmargin: float,
    ymargin: float,
    axis: int = 1,
) -> np.ndarray:
    """Finds candidates not overlapping with lines

    Args:
        lines_xyxy (np.ndarray): line segments
        candidates (np.ndarray): candidate boxes
        xmargin (float): fraction of the x-dimension to use as margins for text boxes
        ymargin (float): fraction of the y-dimension to use as margins for text boxes
        axis (int): If axis set to 0, performs the opposite comparison (lines to boxes)

    Returns:
        np.ndarray: Boolean array of shape (K,) with True for non-overlapping candidates with lines.
    """
    non_intersecting = np.invert(
        np.any(
            np.bitwise_or(
                line_intersect(
                    np.hstack(
                        [
                            candidates[:, 0:1] - xmargin,
                            candidates[:, 1:2] - ymargin,
                            candidates[:, 0:1] - xmargin,
                            candidates[:, 3:] + ymargin,
                        ]
                    ),
                    lines_xyxy,
                ),
                np.bitwise_or(
                    line_intersect(
                        np.hstack(
                            [
                                candidates[:, 0:1] - xmargin,
                                candidates[:, 3:] + ymargin,
                                candidates[:, 2:3] + xmargin,
                                candidates[:, 3:] + ymargin,
                            ]
                        ),
                        lines_xyxy,
                    ),
                    np.bitwise_or(
                        line_intersect(
                            np.hstack(
                                [
                                    candidates[:, 2:3] + xmargin,
                                    candidates[:, 3:] + ymargin,
                                    candidates[:, 2:3] + xmargin,
                                    candidates[:, 1:2] - ymargin,
                                ]
                            ),
                            lines_xyxy,
                        ),
                        line_intersect(
                            np.hstack(
                                [
                                    candidates[:, 2:3] + xmargin,
                                    candidates[:, 1:2] - ymargin,
                                    candidates[:, 0:1] - xmargin,
                                    candidates[:, 1:2] - ymargin,
                                ]
                            ),
                            lines_xyxy,
                        ),
                    ),
                ),
            ),
            axis=axis,
        )
    )

    non_inside = np.invert(
        np.any(
            np.bitwise_and(
                candidates[:, 0][:, None] - xmargin < lines_xyxy[:, 0],
                np.bitwise_and(
                    candidates[:, 1][:, None] - ymargin < lines_xyxy[:, 1],
                    np.bitwise_and(
                        candidates[:, 2][:, None] + xmargin > lines_xyxy[:, 0],
                        np.bitwise_and(
                            candidates[:, 3][:, None] + ymargin > lines_xyxy[:, 1],
                            np.bitwise_and(
                                candidates[:, 0][:, None] - xmargin < lines_xyxy[:, 2],
                                np.bitwise_and(
                                    candidates[:, 1][:, None] - ymargin
                                    < lines_xyxy[:, 3],
                                    np.bitwise_and(
                                        candidates[:, 2][:, None] + xmargin
                                        > lines_xyxy[:, 2],
                                        candidates[:, 3][:, None] + ymargin
                                        > lines_xyxy[:, 3],
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            axis=axis,
        )
    )
    return np.bitwise_and(non_intersecting, non_inside)


def line_intersect(cand_xyxy: np.ndarray, lines_xyxy: np.ndarray) -> np.ndarray:
    """Checks if line segments intersect for all line segments and candidates.

    Args:
        cand_xyxy (np.ndarray): line segments in candidates
        lines_xyxy (np.ndarray): line segments plotted

    Returns:
        np.ndarray: Boolean array with True for non-overlapping candidate segments with line segments.
    """
    intersects = np.bitwise_and(
        ccw(cand_xyxy[:, :2], lines_xyxy[:, :2], lines_xyxy[:, 2:], False)
        != ccw(cand_xyxy[:, 2:], lines_xyxy[:, :2], lines_xyxy[:, 2:], False),
        ccw(cand_xyxy[:, :2], cand_xyxy[:, 2:], lines_xyxy[:, :2], True)
        != ccw(cand_xyxy[:, :2], cand_xyxy[:, 2:], lines_xyxy[:, 2:], True),
    )
    return intersects


def ccw(x1y1: np.ndarray, x2y2: np.ndarray, x3y3: np.ndarray, cand: bool) -> np.ndarray:
    """CCW used in line intersect

    Args:
        x1y1 (np.ndarray):
        x2y2 (np.ndarray):
        x3y3 (np.ndarray):
        cand (bool): using candidate positions (different broadcasting)

    Returns:
        np.ndarray:
    """
    if cand:
        return (
            (-(x1y1[:, 1][:, None] - x3y3[:, 1]))
            * np.repeat(x2y2[:, 0:1] - x1y1[:, 0:1], x3y3.shape[0], axis=1)
        ) > (
            np.repeat(x2y2[:, 1:2] - x1y1[:, 1:2], x3y3.shape[0], axis=1)
            * (-(x1y1[:, 0][:, None] - x3y3[:, 0]))
        )
    return (
        (-(x1y1[:, 1][:, None] - x3y3[:, 1])) * (-(x1y1[:, 0][:, None] - x2y2[:, 0]))
    ) > ((-(x1y1[:, 1][:, None] - x2y2[:, 1])) * (-(x1y1[:, 0][:, None] - x3y3[:, 0])))


def non_overlapping_with_boxes(
    box_arr: np.ndarray, candidates: np.ndarray, xmargin: float, ymargin: float
) -> np.ndarray:
    """Finds candidates not overlapping with allocated boxes.

    Args:
        box_arr (np.ndarray): array with allocated boxes
        candidates (np.ndarray): candidate boxes
        xmargin (float): fraction of the x-dimension to use as margins for text boxes
        ymargin (float): fraction of the y-dimension to use as margins for text boxes

    Returns:
        np.ndarray: Boolean array of shape (K,) with True for non-overlapping candidates with boxes.
    """
    return np.invert(
        np.any(
            np.invert(
                np.bitwise_or(
                    candidates[:, 0][:, None] - xmargin > box_arr[:, 2],
                    np.bitwise_or(
                        candidates[:, 2][:, None] + xmargin < box_arr[:, 0],
                        np.bitwise_or(
                            candidates[:, 1][:, None] - ymargin > box_arr[:, 3],
                            candidates[:, 3][:, None] + ymargin < box_arr[:, 1],
                        ),
                    ),
                )
            ),
            axis=1,
        )
    )


def inside_plot(
    xmin_bound: float,
    ymin_bound: float,
    xmax_bound: float,
    ymax_bound: float,
    candidates: np.ndarray,
) -> np.ndarray:
    """Finds candidates that are inside the plot bounds

    Args:
        xmin_bound (float):
        ymin_bound (float):
        xmax_bound (float):
        ymax_bound (float):
        candidates (np.ndarray): candidate boxes

    Returns:
        np.ndarray: Boolean array of shape (K,) with True for non-overlapping candidates with boxes.
    """
    return np.invert(
        np.bitwise_or(
            candidates[:, 0] < xmin_bound,
            np.bitwise_or(
                candidates[:, 1] < ymin_bound,
                np.bitwise_or(
                    candidates[:, 2] > xmax_bound, candidates[:, 3] > ymax_bound
                ),
            ),
        )
    )

import numpy as np


def non_overlapping_with_points(
    scatter_xy: np.ndarray, candidates: np.ndarray, xfrac: float, yfrac: float
) -> np.ndarray:
    # """
    # Parameters
    # ----------
    # pointarr : np.ndarray
    #     Array of shape (N,2) containing coordinates for all scatter-points
    # candidates : np.ndarray
    #     Array of shape (K,4) with K candidate patches
    # xfrac : float
    #     fraction of the x-dimension to use as margins for text bboxes
    # yfrac : float
    #     fraction of the y-dimension to use as margins for text bboxes

    # Returns
    # -------
    # np.array
    # Boolean array of shape (K,) with True for non-overlapping candidates with points

    # """
    return np.invert(
        np.bitwise_or.reduce(
            np.bitwise_and(
                candidates[:, 0][:, None] - xfrac < scatter_xy[:, 0],
                np.bitwise_and(
                    candidates[:, 2][:, None] + xfrac > scatter_xy[:, 0],
                    np.bitwise_and(
                        candidates[:, 1][:, None] - yfrac < scatter_xy[:, 1],
                        candidates[:, 3][:, None] + yfrac > scatter_xy[:, 1],
                    ),
                ),
            ),
            axis=1,
        )
    )


def ccw(x1y1, x2y2, x3y3, cand):
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


def line_intersect(cand_xyxy, lines_xyxy):
    intersects = np.bitwise_and(
        ccw(cand_xyxy[:, :2], lines_xyxy[:, :2], lines_xyxy[:, 2:], False)
        != ccw(cand_xyxy[:, 2:], lines_xyxy[:, :2], lines_xyxy[:, 2:], False),
        ccw(cand_xyxy[:, :2], cand_xyxy[:, 2:], lines_xyxy[:, :2], True)
        != ccw(cand_xyxy[:, :2], cand_xyxy[:, 2:], lines_xyxy[:, 2:], True),
    )
    return intersects


def non_overlapping_with_lines(
    lines_xyxy: np.ndarray, candidates: np.ndarray, xfrac: float, yfrac: float
) -> np.ndarray:
    non_intersecting = np.invert(
        np.any(
            np.bitwise_or(
                line_intersect(
                    np.hstack(
                        [
                            candidates[:, 0:1] - xfrac,
                            candidates[:, 1:2] - yfrac,
                            candidates[:, 0:1] - xfrac,
                            candidates[:, 3:] + yfrac,
                        ]
                    ),
                    lines_xyxy,
                ),
                np.bitwise_or(
                    line_intersect(
                        np.hstack(
                            [
                                candidates[:, 0:1] - xfrac,
                                candidates[:, 3:] + yfrac,
                                candidates[:, 2:3] + xfrac,
                                candidates[:, 3:] + yfrac,
                            ]
                        ),
                        lines_xyxy,
                    ),
                    np.bitwise_or(
                        line_intersect(
                            np.hstack(
                                [
                                    candidates[:, 2:3] + xfrac,
                                    candidates[:, 3:] + yfrac,
                                    candidates[:, 2:3] + xfrac,
                                    candidates[:, 1:2] - yfrac,
                                ]
                            ),
                            lines_xyxy,
                        ),
                        line_intersect(
                            np.hstack(
                                [
                                    candidates[:, 2:3] + xfrac,
                                    candidates[:, 1:2] - yfrac,
                                    candidates[:, 0:1] - xfrac,
                                    candidates[:, 1:2] - yfrac,
                                ]
                            ),
                            lines_xyxy,
                        ),
                    ),
                ),
            ),
            axis=1,
        )
    )

    non_inside = np.invert(
        np.any(
            np.bitwise_and(
                candidates[:, 0][:, None] - xfrac < lines_xyxy[:, 0],
                np.bitwise_and(
                    candidates[:, 1][:, None] - yfrac < lines_xyxy[:, 1],
                    np.bitwise_and(
                        candidates[:, 2][:, None] + xfrac > lines_xyxy[:, 0],
                        np.bitwise_and(
                            candidates[:, 3][:, None] + yfrac > lines_xyxy[:, 1],
                            np.bitwise_and(
                                candidates[:, 0][:, None] - xfrac < lines_xyxy[:, 2],
                                np.bitwise_and(
                                    candidates[:, 1][:, None] - yfrac
                                    < lines_xyxy[:, 3],
                                    np.bitwise_and(
                                        candidates[:, 2][:, None] + xfrac
                                        > lines_xyxy[:, 2],
                                        candidates[:, 3][:, None] + yfrac
                                        > lines_xyxy[:, 3],
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            axis=1,
        )
    )
    return np.bitwise_and(non_intersecting, non_inside)


def non_overlapping_with_rectangles(rectangle_arr, candidates, xfrac, yfrac):
    # """
    # Parameters
    # ----------
    # rectangle_arr : np.ndarray
    #     Array of shape (N,4) containing patches of all added patches so far
    # candidates : np.ndarray
    #     Array of shape (K,4) with K candidate patches
    # xfrac : float
    #     fraction of the x-dimension to use as margins for text bboxes
    # yfrac : float
    #     fraction of the y-dimension to use as margins for text bboxes

    # Returns
    # -------
    # np.array
    # Boolean array of shape (K,) with True for non-overlapping candidates with points

    # """
    return np.invert(
        np.any(
            np.invert(
                np.bitwise_or(
                    candidates[:, 0][:, None] - xfrac > rectangle_arr[:, 2],
                    np.bitwise_or(
                        candidates[:, 2][:, None] + xfrac < rectangle_arr[:, 0],
                        np.bitwise_or(
                            candidates[:, 1][:, None] - yfrac > rectangle_arr[:, 3],
                            candidates[:, 3][:, None] + yfrac < rectangle_arr[:, 1],
                        ),
                    ),
                )
            ),
            axis=1,
        )
    )


def inside_plot(xmin_bound, ymin_bound, xmax_bound, ymax_bound, candidates):
    # """
    # Parameters
    # ----------
    # xmin_bound : float
    # ymin_bound : float
    # xmax_bound : float
    # ymax_bound : float
    # candidates : np.ndarray
    #     Array of shape (K,4) with K candidate patches

    # Returns
    # -------
    # np.array
    # Boolean array of shape (K,) with True for non-overlapping candidates with points

    # """
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

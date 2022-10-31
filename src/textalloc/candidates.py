import numpy as np


def generate_candidates(
    w: float,
    h: float,
    x: float,
    y: float,
    xfrac: float,
    yfrac: float,
    nbr_candidates: int,
) -> np.ndarray:
    """Generates 36 candidate boxes

    Args:
        w (float): width of box
        h (float): height of box
        x (float): xmin of box
        y (float): ymin of box
        xfrac (float): fraction of the x-dimension to use as margins for text bboxes
        yfrac (float): fraction of the y-dimension to use as margins for text bboxes
        nbr_candidates (int): nbr of candidates to use. If <1 or >36 uses all 36

    Returns:
        np.ndarray: candidate boxes array
    """
    candidates = np.array(
        [
            [x + xfrac, y + yfrac, x + w + xfrac, y + h + yfrac],  # upper right side
            [x - w - xfrac, y + yfrac, x - xfrac, y + h + yfrac],  # upper left side
            [x - w - xfrac, y - h - yfrac, x - xfrac, y - yfrac],  # lower left side
            [x + xfrac, y - h - yfrac, x + w + xfrac, y - yfrac],  # lower right side
            [x - w - xfrac, y - h / 2, x - xfrac, y + h / 2],  # left side
            [x + xfrac, y - h / 2, x + w + xfrac, y + h / 2],  # right side
            [x - w / 2, y + yfrac, x + w / 2, y + h + yfrac],  # above
            [x - w / 2, y - h - yfrac, x + w / 2, y - yfrac],  # below
            [x - 3 * w / 4, y + yfrac, x + w / 4, y + h + yfrac],  # above left
            [x - w / 4, y + yfrac, x + 3 * w / 4, y + h + yfrac],  # above right
            [x - 3 * w / 4, y - h - yfrac, x + w / 4, y - yfrac],  # below left
            [x - w / 4, y - h - yfrac, x + 3 * w / 4, y - yfrac],  # below right
            # We move all points a bit further from the target
            [
                x + 2 * xfrac,
                y + 2 * yfrac,
                x + w + 2 * xfrac,
                y + h + 2 * yfrac,
            ],  # upper right side
            [
                x - w - 2 * xfrac,
                y + 2 * yfrac,
                x - 2 * xfrac,
                y + h + 2 * yfrac,
            ],  # upper left side
            [
                x - w - 2 * xfrac,
                y - h - 2 * yfrac,
                x - 2 * xfrac,
                y - 2 * yfrac,
            ],  # lower left side
            [
                x + 2 * xfrac,
                y - h - 2 * yfrac,
                x + w + 2 * xfrac,
                y - 2 * yfrac,
            ],  # lower right side
            [x - w - 2 * xfrac, y - h / 2, x - 2 * xfrac, y + h / 2],  # left side
            [x + 2 * xfrac, y - h / 2, x + w + 2 * xfrac, y + h / 2],  # right side
            [x - w / 2, y + 2 * yfrac, x + w / 2, y + h + 2 * yfrac],  # above
            [x - w / 2, y - h - 2 * yfrac, x + w / 2, y - 2 * yfrac],  # below
            [x - 3 * w / 4, y + 2 * yfrac, x + w / 4, y + h + 2 * yfrac],  # above left
            [x - w / 4, y + 2 * yfrac, x + 3 * w / 4, y + h + 2 * yfrac],  # above right
            [x - 3 * w / 4, y - h - 2 * yfrac, x + w / 4, y - 2 * yfrac],  # below left
            [x - w / 4, y - h - 2 * yfrac, x + 3 * w / 4, y - 2 * yfrac],  # below right
            [
                x + 3 * xfrac,
                y + 3 * yfrac,
                x + w + 3 * xfrac,
                y + h + 3 * yfrac,
            ],  # upper right side
            [
                x - w - 3 * xfrac,
                y + 3 * yfrac,
                x - 3 * xfrac,
                y + h + 3 * yfrac,
            ],  # upper left side
            [
                x - w - 3 * xfrac,
                y - h - 3 * yfrac,
                x - 3 * xfrac,
                y - 3 * yfrac,
            ],  # lower left side
            [
                x + 3 * xfrac,
                y - h - 3 * yfrac,
                x + w + 3 * xfrac,
                y - 3 * yfrac,
            ],  # lower right side
            [x - w - 3 * xfrac, y - h / 2, x - 3 * xfrac, y + h / 2],  # left side
            [x + 3 * xfrac, y - h / 2, x + w + 3 * xfrac, y + h / 2],  # right side
            [x - w / 2, y + 3 * yfrac, x + w / 2, y + h + 3 * yfrac],  # above
            [x - w / 2, y - h - 3 * yfrac, x + w / 2, y - 3 * yfrac],  # below
            [x - 3 * w / 4, y + 3 * yfrac, x + w / 4, y + h + 3 * yfrac],  # above left
            [x - w / 4, y + 3 * yfrac, x + 3 * w / 4, y + h + 3 * yfrac],  # above right
            [x - 3 * w / 4, y - h - 3 * yfrac, x + w / 4, y - 3 * yfrac],  # below left
            [x - w / 4, y - h - 3 * yfrac, x + 3 * w / 4, y - 3 * yfrac],  # below right
        ]
    )
    if nbr_candidates > 0 and nbr_candidates < len(candidates):
        return candidates[:nbr_candidates, :]
    return candidates

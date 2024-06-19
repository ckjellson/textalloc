import math
import numpy as np


direction_to_dir = {
    "north": (0, 1),
    "south": (0, -1),
    "east": (1, 0),
    "west": (-1, 0),
    "northeast": (1, 1),
    "northwest": (-1, 1),
    "southest": (1, -1),
    "southwest": (-1, -1),
}


def generate_candidates(
    w: float,
    h: float,
    x: float,
    y: float,
    xmindistance: float,
    ymindistance: float,
    xmaxdistance: float,
    ymaxdistance: float,
    nbr_candidates: int,
    scatter_size: float,
    direction: str,
) -> np.ndarray:
    """Generates candidate boxes

    Args:
        w (float): width of box
        h (float): height of box
        x (float): xmin of box
        y (float): ymin of box
        xmindistance (float): fraction of the x-dimension to use as margins for text bboxes
        ymindistance (float): fraction of the y-dimension to use as margins for text bboxes
        xmaxdistance (float): fraction of the x-dimension to use as max distance for text bboxes
        ymaxdistance (float): fraction of the y-dimension to use as max distance for text bboxes
        nbr_candidates (int): nbr of candidates to use. If <1 or >36 uses all 36
        scatter_size (float): size of scattered text objects.
        direction (str): set preferred loaction of the boxes.

    Returns:
        np.ndarray: candidate boxes array
    """
    xmindistance += scatter_size
    ymindistance += scatter_size
    xmaxdistance += scatter_size
    ymaxdistance += scatter_size

    candidates1 = None
    if direction is not None:
        dir = direction_to_dir[direction]
        dir_cands = []
        for i in range(1, 10):
            if dir[0] == -1:
                x_ = x - w - xmindistance * i
            elif dir[0] == 0:
                x_ = x - w / 2
            elif dir[0] == 1:
                x_ = x + xmindistance * i
            if dir[1] == -1:
                y_ = y - h - ymindistance * i
            elif dir[1] == 0:
                y_ = y - h / 2
            elif dir[1] == 1:
                y_ = y + ymindistance * i
            dir_cands.append([x_, y_, x_ + w, y_ + h])
        candidates1 = np.array(dir_cands)

    candidates = np.array(
        [
            [
                x + xmindistance,
                y + ymindistance,
                x + w + xmindistance,
                y + h + ymindistance,
            ],  # upper right side
            [
                x - w - xmindistance,
                y + ymindistance,
                x - xmindistance,
                y + h + ymindistance,
            ],  # upper left side
            [
                x - w - xmindistance,
                y - h - ymindistance,
                x - xmindistance,
                y - ymindistance,
            ],  # lower left side
            [
                x + xmindistance,
                y - h - ymindistance,
                x + w + xmindistance,
                y - ymindistance,
            ],  # lower right side
            [x - w - xmindistance, y - h / 2, x - xmindistance, y + h / 2],  # left side
            [
                x + xmindistance,
                y - h / 2,
                x + w + xmindistance,
                y + h / 2,
            ],  # right side
            [x - w / 2, y + ymindistance, x + w / 2, y + h + ymindistance],  # above
            [x - w / 2, y - h - ymindistance, x + w / 2, y - ymindistance],  # below
            [
                x - 3 * w / 4,
                y + ymindistance,
                x + w / 4,
                y + h + ymindistance,
            ],  # above left
            [
                x - w / 4,
                y + ymindistance,
                x + 3 * w / 4,
                y + h + ymindistance,
            ],  # above right
            [
                x - 3 * w / 4,
                y - h - ymindistance,
                x + w / 4,
                y - ymindistance,
            ],  # below left
            [
                x - w / 4,
                y - h - ymindistance,
                x + 3 * w / 4,
                y - ymindistance,
            ],  # below right
            # We move all points a bit further from the target
        ]
    )
    if direction is not None:
        candidates = np.vstack([candidates1, candidates])
    if nbr_candidates > candidates.shape[0]:
        area = xmaxdistance * ymaxdistance - xmindistance*ymindistance
        sampling_size = np.sqrt(area/nbr_candidates)

        n_samples_x = math.ceil(xmaxdistance/sampling_size)
        n_samples_y = math.ceil(ymaxdistance/sampling_size)

        grid_x, grid_y = np.meshgrid(np.linspace(-xmaxdistance, xmaxdistance, n_samples_x),
                                     np.linspace(-ymaxdistance, ymaxdistance, n_samples_y),
                                     indexing="xy")
        grid = np.stack((grid_x, grid_y), axis=-1).reshape(-1, 2)
        grid = grid[np.logical_or(np.abs(grid[:, 0]) > xmindistance,
                                  np.abs(grid[:, 1]) > ymindistance)]
        grid = grid[np.argsort(grid[:, 0]**2 + grid[:, 1]**2)]

        candidates2 = np.stack((grid[:, 0] + x - w / 2,
                                grid[:, 1] + y - h / 2,
                                grid[:, 0] + x + w / 2,
                                grid[:, 1] + y + h / 2), axis=-1)

        candidates2 = candidates2[:nbr_candidates - candidates.shape[0]]

        candidates = np.vstack([candidates, candidates2])

    if direction is not None:
        if direction == "south":
            candidates = candidates[(candidates[:, 1] < y) & (candidates[:, 3] < y), :]
        elif direction == "north":
            candidates = candidates[(candidates[:, 1] > y) & (candidates[:, 3] > y), :]
        elif direction == "west":
            candidates = candidates[(candidates[:, 0] < x) & (candidates[:, 2] < x), :]
        elif direction == "east":
            candidates = candidates[(candidates[:, 0] > x) & (candidates[:, 2] > x), :]
        elif direction == "southwest":
            candidates = candidates[
                (candidates[:, 1] < y)
                & (candidates[:, 3] < y)
                & (candidates[:, 0] < x)
                & (candidates[:, 2] < x),
                :,
            ]
        elif direction == "southeast":
            candidates = candidates[
                (candidates[:, 1] < y)
                & (candidates[:, 3] < y)
                & (candidates[:, 0] > x)
                & (candidates[:, 2] > x),
                :,
            ]
        elif direction == "northwest":
            candidates = candidates[
                (candidates[:, 1] > y)
                & (candidates[:, 3] > y)
                & (candidates[:, 0] < x)
                & (candidates[:, 2] < x),
                :,
            ]
        elif direction == "northeast":
            candidates = candidates[
                (candidates[:, 1] > y)
                & (candidates[:, 3] > y)
                & (candidates[:, 0] > x)
                & (candidates[:, 2] > x),
                :,
            ]
    return candidates

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

import math
import numpy as np
from typing import List


direction_to_dir = {
    "north": (0, 1),
    "south": (0, -1),
    "east": (1, 0),
    "west": (-1, 0),
    "northeast": (1, 1),
    "northwest": (-1, 1),
    "southeast": (1, -1),
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
    direction: List[str],
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
        candidates1 = []
        for i in range(1, 10):
            i_cands = []
            for d in direction:
                dir = direction_to_dir[d]
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
                i_cands.append([x_, y_, x_ + w, y_ + h])
            candidates1.append(np.array(i_cands))
        candidates1 = np.vstack(candidates1)

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
        area = xmaxdistance * ymaxdistance - xmindistance * ymindistance
        sampling_size = np.sqrt(area / nbr_candidates)

        n_samples_x = math.ceil(xmaxdistance / sampling_size)
        n_samples_y = math.ceil(ymaxdistance / sampling_size)

        grid_x, grid_y = np.meshgrid(
            np.linspace(-xmaxdistance, xmaxdistance, n_samples_x),
            np.linspace(-ymaxdistance, ymaxdistance, n_samples_y),
            indexing="xy",
        )
        grid = np.stack((grid_x, grid_y), axis=-1).reshape(-1, 2)
        grid = grid[
            np.logical_or(
                np.abs(grid[:, 0]) > xmindistance, np.abs(grid[:, 1]) > ymindistance
            )
        ]
        grid = grid[np.argsort(grid[:, 0] ** 2 + grid[:, 1] ** 2)]

        candidates2 = np.stack(
            (
                grid[:, 0] + x - w / 2,
                grid[:, 1] + y - h / 2,
                grid[:, 0] + x + w / 2,
                grid[:, 1] + y + h / 2,
            ),
            axis=-1,
        )

        candidates2 = candidates2[: nbr_candidates - candidates.shape[0]]

        candidates = np.vstack([candidates, candidates2])

    if direction is not None:
        final_candidates = []
        mask = candidates[:, 1] > 1e10
        for d in direction:
            if d == "south":
                mask[(candidates[:, 1] < y) & (candidates[:, 3] < y)] = True
            elif d == "north":
                mask[(candidates[:, 1] > y) & (candidates[:, 3] > y)] = True
            elif d == "west":
                mask[(candidates[:, 0] < x) & (candidates[:, 2] < x)] = True
            elif d == "east":
                mask[(candidates[:, 0] > x) & (candidates[:, 2] > x)] = True
            elif d == "southwest":
                mask[
                    (candidates[:, 1] < y)
                    & (candidates[:, 3] < y)
                    & (candidates[:, 0] < x)
                    & (candidates[:, 2] < x)] = True
            elif d == "southeast":
                mask[
                    (candidates[:, 1] < y)
                    & (candidates[:, 3] < y)
                    & (candidates[:, 0] > x)
                    & (candidates[:, 2] > x)] = True
            elif d == "northwest":
                mask[
                    (candidates[:, 1] > y)
                    & (candidates[:, 3] > y)
                    & (candidates[:, 0] < x)
                    & (candidates[:, 2] < x)] = True
            elif d == "northeast":
                mask[
                    (candidates[:, 1] > y)
                    & (candidates[:, 3] > y)
                    & (candidates[:, 0] > x)
                    & (candidates[:, 2] > x)] = True
        candidates = candidates[mask,:]
    return candidates

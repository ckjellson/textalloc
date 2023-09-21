from tqdm import tqdm
from textalloc.non_overlapping_boxes import get_non_overlapping_boxes
import numpy as np
import time
from typing import List, Tuple, Union


def allocate_text(
    fig,
    ax,
    x: Union[np.ndarray, List[float]],
    y: Union[np.ndarray, List[float]],
    text_list: List[str],
    x_scatter: Union[np.ndarray, List[float]] = None,
    y_scatter: Union[np.ndarray, List[float]] = None,
    x_lines: List[Union[np.ndarray, List[float]]] = None,
    y_lines: List[Union[np.ndarray, List[float]]] = None,
    scatter_sizes: List[Union[np.ndarray, List[float]]] = None,
    text_scatter_sizes: List[Union[np.ndarray, List[float]]] = None,
    textsize: Union[int, List[int]] = 10,
    margin: float = 0.008,
    min_distance: float = 0.013,
    max_distance: float = 0.2,
    verbose: bool = False,
    draw_lines: bool = True,
    linecolor: str = "r",
    draw_all: bool = True,
    nbr_candidates: int = 200,
    linewidth: float = 1,
    textcolor: Union[str, List[str]] = "k",
    seed: int = 0,
    **kwargs,
):
    """Main function of allocating text-boxes in matplotlib plot

    Args:
        fig (_type_): matplotlib figure used for rendering textbox-sizes.
        ax (_type_): matplotlib axes used for plotting.
        x (Union[np.ndarray, List[float]]): x-coordinates of texts 1d array/list.
        y (Union[np.ndarray, List[float]]): y-coordinates of texts 1d array/list.
        text_list (List[str]): list of texts.
        x_scatter (Union[np.ndarray, List[float]], optional): x-coordinates of all scattered points in plot 1d array/list. Defaults to None.
        y_scatter (Union[np.ndarray, List[float]], optional): y-coordinates of all scattered points in plot 1d array/list. Defaults to None.
        x_lines (List[Union[np.ndarray, List[float]]], optional): x-coordinates of all lines in plot list of 1d arrays/lists. Defaults to None.
        y_lines (List[Union[np.ndarray, List[float]]], optional): y-coordinates of all lines in plot list of 1d arrays/lists. Defaults to None.
        scatter_sizes (List[Union[np.ndarray, List[float]]], optional): sizes of all scattered objects in plot list of 1d arrays/lists. Defaults to None.
        text_scatter_sizes (List[Union[np.ndarray, List[float]]], optional): sizes of text scattered objects in plot list of 1d arrays/lists. Defaults to None.
        textsize (Union[int, List[int]], optional): size of text. Defaults to 10.
        margin (float, optional): parameter for margins between objects. Increase for larger margins to points and lines. Defaults to 0.0.
        min_distance (float, optional): parameter for min distance between text and origin. Defaults to 0.015.
        max_distance (float, optional): parameter for max distance between text and origin. Defaults to 0.2.
        verbose (bool, optional): prints progress using tqdm. Defaults to False.
        draw_lines (bool, optional): draws lines from original points to textboxes. Defaults to True.
        linecolor (str, optional): color code of the lines between points and text-boxes. Defaults to "r".
        draw_all (bool, optional): Draws all texts after allocating as many as possible despit overlap. Defaults to True.
        nbr_candidates (int, optional): Sets the number of candidates used. Defaults to 200.
        linewidth (float, optional): width of line. Defaults to 1.
        textcolor (Union[str, List[str]], optional): color code of the text. Defaults to "k".
        seed (int, optional): seeds order of text allocations. Defaults to 0.
        **kwargs (): kwargs for the plt.text() call.
    """
    t0 = time.time()
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()

    # Ensure good inputs
    assert len(x) == len(y)
    x = np.array(x)
    y = np.array(y)
    if scatter_sizes is not None:
        scatter_sizes = np.array(scatter_sizes)
    if text_scatter_sizes is not None:
        text_scatter_sizes = np.array(text_scatter_sizes)
    if x_scatter is not None:
        assert y_scatter is not None
    if y_scatter is not None:
        assert x_scatter is not None
        assert len(y_scatter) == len(x_scatter)
        x_scatter = np.array(x_scatter)
        y_scatter = np.array(y_scatter)
    if x_lines is not None:
        assert y_lines is not None
    if y_lines is not None:
        assert x_lines is not None
        assert all(
            [len(x_line) == len(y_line) for x_line, y_line in zip(x_lines, y_lines)]
        )
        x_lines = [np.array(x_line) for x_line in x_lines]
        y_lines = [np.array(y_line) for y_line in y_lines]
    assert min_distance <= max_distance
    assert min_distance >= margin
    if type(textsize) is not int:
        assert len(textsize) == len(x)
    else:
        textsize = [textsize for _ in range(len(x))]
    if type(textcolor) is not str:
        assert len(textcolor) == len(x)
    else:
        textcolor = [textcolor for _ in range(len(x))]

    # Seed
    if seed > 0:
        randinds = np.arange(x.shape[0])
        np.random.seed(seed)
        np.random.shuffle(randinds)
        text_list = [text_list[i] for i in randinds]
        x = x[randinds]
        y = y[randinds]
        if text_scatter_sizes is not None:
            text_scatter_sizes = text_scatter_sizes[randinds]

    # Create boxes in original plot
    if verbose:
        print("Creating boxes")
    original_boxes = []
    for x_coord, y_coord, s, ts in tqdm(
        zip(x, y, text_list, textsize), disable=not verbose
    ):
        ann = ax.text(x_coord, y_coord, s, size=ts)
        box = ax.transData.inverted().transform(
            ann.get_tightbbox(fig.canvas.get_renderer())
        )
        w, h = box[1][0] - box[0][0], box[1][1] - box[0][1]
        original_boxes.append((x_coord, y_coord, w, h, s))
        ann.remove()

    # Process extracted textboxes
    if verbose:
        print("Processing")
    if x_scatter is None:
        scatterxy = None
    else:
        scatterxy = np.transpose(np.vstack([x_scatter, y_scatter]))
    if x_lines is None:
        lines_xyxy = None
    else:
        lines_xyxy = lines_to_segments(x_lines, y_lines)
    non_overlapping_boxes, overlapping_boxes_inds = get_non_overlapping_boxes(
        original_boxes,
        xlims,
        ylims,
        margin,
        min_distance,
        max_distance,
        verbose,
        nbr_candidates,
        draw_all,
        scatter_xy=scatterxy,
        lines_xyxy=lines_xyxy,
        scatter_sizes=scatter_sizes,
        text_scatter_sizes=text_scatter_sizes,
    )

    # Plot once again
    if verbose:
        print("Plotting")
    if draw_lines:
        for x_coord, y_coord, w, h, s, ind in non_overlapping_boxes:
            x_near, y_near = find_nearest_point_on_box(
                x_coord, y_coord, w, h, x[ind], y[ind]
            )
            if x_near is not None:
                ax.plot(
                    [x[ind], x_near], [y[ind], y_near], linewidth=linewidth, c=linecolor
                )
    for x_coord, y_coord, w, h, s, ind in non_overlapping_boxes:
        ax.text(x_coord, y_coord, s, size=textsize[ind], c=textcolor[ind], **kwargs)

    if draw_all:
        for ind in overlapping_boxes_inds:
            ax.text(
                x[ind],
                y[ind],
                text_list[ind],
                size=textsize[ind],
                c=textcolor[ind],
                **kwargs,
            )

    if verbose:
        print(f"Finished in {time.time()-t0}s")


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


def lines_to_segments(
    x_lines: List[np.ndarray],
    y_lines: List[np.ndarray],
) -> np.ndarray:
    """Sets up

    Args:
        x_lines (List[np.ndarray]): x-coordinates of all lines in plot list of 1d arrays
        y_lines (List[np.ndarray]): y-coordinates of all lines in plot list of 1d arrays

    Returns:
        np.ndarray: 2d array of line segments
    """
    assert len(x_lines) == len(y_lines)
    n_x_segments = np.sum([len(line_x) - 1 for line_x in x_lines])
    n_y_segments = np.sum([len(line_y) - 1 for line_y in y_lines])
    assert n_x_segments == n_y_segments
    lines_xyxy = np.zeros((n_x_segments, 4))
    iter = 0
    for line_x, line_y in zip(x_lines, y_lines):
        for i in range(len(line_x) - 1):
            lines_xyxy[iter, :] = [line_x[i], line_y[i], line_x[i + 1], line_y[i + 1]]
            iter += 1
    return lines_xyxy

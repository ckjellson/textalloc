from tqdm import tqdm
from textalloc.find_non_overlapping import get_non_overlapping_patches
import numpy as np
import time


def allocate_text(
    fig,
    ax,
    x,
    y,
    text_list,
    xlims,
    ylims,
    x_scatter=None,
    y_scatter=None,
    x_lines=None,
    y_lines=None,
    textsize=10,
    distance_margin_fraction=0.015,
    verbose=False,
    draw_lines=False,
    linecolor="k",
):
    t0 = time.time()

    if verbose:
        print("Creating patches")
    original_patches = []
    for x_coord, y_coord, s in tqdm(zip(x, y, text_list), disable=not verbose):
        ann = ax.text(x_coord, y_coord, s, size=textsize)
        patch = ax.transData.inverted().transform(
            ann.get_tightbbox(fig.canvas.get_renderer())
        )
        w, h = patch[1][0] - patch[0][0], patch[1][1] - patch[0][1]
        original_patches.append((x_coord, y_coord, w, h, s))
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
    non_overlapping_patches = get_non_overlapping_patches(
        original_patches,
        xlims,
        ylims,
        distance_margin_fraction,
        verbose,
        scatter_xy=scatterxy,
        lines_xyxy=lines_xyxy,
    )

    # Plot once again
    if verbose:
        print("Plotting")
    for x_coord, y_coord, w, h, s, ind in non_overlapping_patches:
        ax.text(x_coord, y_coord, s, size=textsize)
        if draw_lines:
            x_near, y_near = find_nearest_point_on_rectangle(
                x_coord, y_coord, w, h, x[ind], y[ind]
            )
            if x_near is not None:
                ax.plot([x[ind], x_near], [y[ind], y_near], linewidth=1, c=linecolor)

    if verbose:
        print(f"Finished in {time.time()-t0}s")
    return fig, ax


def find_nearest_point_on_rectangle(xmin, ymin, w, h, x, y):
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


def lines_to_segments(x_lines, y_lines):
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

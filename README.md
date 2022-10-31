# textalloc - Efficient Text Allocation in matplotlib using NumPy Broadcasting

Original|textalloc
:-------------------------:|:-------------------------:
![](images/scattertext_before.png)|![](images/scattertext_after.png)
<div align="center">
Scatterplot design from scattertext (https://github.com/JasonKessler/scattertext)
</div>

# Quick-start

## Installation

```
pip install textalloc
```

## Using textalloc

The code below generates the plot to the right:

```
import textalloc as ta
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
x, y = np.random.random((2,30))
fig, ax = plt.subplots()
ax.scatter(x, y, c='b')
text_list = [f'Text{i}' for i in range(len(x))]
ta.allocate_text(fig, ax, x, y, text_list, ax.get_xlim(), ax.get_ylim(), x_scatter=x, y_scatter=y, draw_lines=True, distance_margin_fraction=0.025)
plt.show()
```

Original|textalloc
:-------------------------:|:-------------------------:
![](images/scatter_before.png)|![](images/scatter_after.png)

## Parameters

**fig**:
- matplotlib figure used for rendering textbox-sizes.

**ax**:
- matplotlib axes used for plotting.

**x** <em>(Union[np.ndarray, List[float]])</em>:
- x-coordinates of texts 1d array/list.

**y** <em>(Union[np.ndarray, List[float]])</em>:
- y-coordinates of texts 1d array/list.

**text_list** <em>(List[str])</em>:
- list of texts.

**xlims** <em>(Tuple[float, float])</em>:
- x-limits of plot gotten from ax.get_xlim() (avoids issues with using the function from various interfaces).

**ylims** <em>(Tuple[float, float])</em>:
- y-limits of plot gotten from ax.get_ylim() (avoids issues with using the function from various interfaces).

**x_scatter** <em>(Union[np.ndarray, List[float]])</em>:
- x-coordinates of all scattered points in plot 1d array/list.

**y_scatter** <em>(Union[np.ndarray, List[float]])</em>:
- y-coordinates of all scattered points in plot 1d array/list.

**x_lines** <em>(List[Union[np.ndarray, List[float]]])</em>:
- x-coordinates of all lines in plot list of 1d arrays/lists.

**y_lines** <em>(List[Union[np.ndarray, List[float]]])</em>:
- y-coordinates of all lines in plot list of 1d arrays/lists.

**textsize** <em>(int)</em>:
- size of text. Default: 10.

**distance_margin_fraction** (float)</em>:
- parameter for margins between objects. Increase for larger margins to points and lines. Default: 0.015.

**verbose** <em>(bool)</em>:
- prints progress using tqdm. Default: False.

**draw_lines** <em>(bool)</em>:
- draws lines from original points to textboxes. Default: False.

**linecolor** <em>(str)</em>:
- color code of the lines between points and text-boxes. Default: "k".

**draw_all** <em>(bool)</em>:
- Draws all texts after allocating as many as possible despit overlap. Default: False.

**nbr_candidates** <em>(int)</em>:
- Sets the number of candidates used. Default: 0 (all candidates used)


# Implementation and speed

The focus in this implementation is on speed and allocating as many text-boxes as possible into the free space in the plot. This is different to for example the great package adjustText (https://github.com/Phlya/adjustText) which keeps all textboxes and tries to adjust these as optimally as possible (a much harder, but also more computationally expensive problem).

There are three main steps of the algorithm:

For each textbox to be plotted:
1. Generate a large number of candidate box boxes near the original point with size that matches the fontsize.
2. Find the candidates that do not overlap any points, lines, plot boundaries, or already allocated boxes.
3. Allocate the text to the first candidate box with no overlaps.

## Speed

The plot in the top of this Readme was generated in 1.98s on a local laptop, and there are rarely more textboxes that fit into one plot. If the result is still too slow to render, try decreasing `nbr_candidates`.

The speed is greatly improved by usage of numpy broadcasting in all functions for computing overlap (see `textalloc/overlap_functions` and `textalloc/find_non_overlapping`). A simple example from the function `non_overlapping_with_boxes` which checks if the candidate boxes (expanded with xfrac, yfrac to provide a margin) overlap with already allocated boxes:

```
candidates[:, 0][:, None] - xfrac > box_arr[:, 2]
```

The statement compares xmin coordinates of all candidates with xmax coordinates of all allocated boxes resulting in a matrix of shape (N_candidates, N_allocated) due to the use of indexing `[:, None]`.

# Types of overlap supported

textalloc supports avoiding overlap with points, lines, and the plot boundary in addition to other text-boxes. See the example below for a combination of the three:

```
import textalloc as ta
import numpy as np
import matplotlib.pyplot as plt

x_line = np.array([0.0, 0.03192317, 0.04101177, 0.26085659, 0.40261173, 0.42142198, 0.87160195, 1.00349979])
y_line = np.array([0. , 0.2, 0.2, 0.4, 0.8, 0.6, 1. , 1. ])
text_list = ['0', '25', '50', '75', '100', '125', '150', '250']
np.random.seed(0)
x, y = np.random.random((2,100))

fig,ax = plt.subplots(dpi=100)
ax.plot(x_line,y_line,color="black")
ax.scatter(x,y,c="b")
ta.allocate_text(fig, ax, x_line, y_line, text_list, ax.get_xlim(), ax.get_ylim(), x_scatter=x, y_scatter=y, x_lines=[x_line], y_lines=[y_line], draw_lines=True)
plt.show()
```

Original|textalloc
:-------------------------:|:-------------------------:
![](images/scatterlines_before.png)|![](images/scatterlines_after.png)

# Improvements

Future improvements:

- Add support for more pyplot objects.
- Improve allocation when draw_lines=True.
# Clustering And KNN Visual Playground

This project contains two small interactive machine learning demos built with Python and Pygame:

- `clustering_example.py` shows how K-Means clustering groups unlabeled points.
- `knn_example.py` shows how K-Nearest Neighbors predicts a label for a query point using nearby labeled samples.

The goal of the project is to make the ideas visual and easy to explore by clicking, moving the mouse, and changing parameters in real time.

## Technologies Used

- Python 3
- Pygame for the interactive window, drawing, animation, and input handling
- Python standard library modules such as `math`, `random`, `dataclasses`, and `collections`

There is no external machine learning framework in this project. The algorithms are implemented directly in Python so the logic stays easy to read and learn from.

## Project Files

- `clustering_example.py`: interactive K-Means visualization
- `knn_example.py`: interactive KNN visualization

## How K-Means Works In This Project

`clustering_example.py` demonstrates unsupervised learning.

K-Means works like this:

1. You add points to the canvas.
2. The program picks `K` starting centroids.
3. Each point is assigned to the nearest centroid.
4. Each centroid moves to the average position of the points assigned to it.
5. Steps 3 and 4 repeat until the groups stop changing or the iteration limit is reached.

In the visualization:

- Points are the data samples.
- Bright cross-shaped markers are the centroids.
- Colored lines connect each point to its current centroid.
- The centroid positions animate as the algorithm updates.

## How KNN Works In This Project

`knn_example.py` demonstrates supervised learning.

KNN works like this:

1. You create labeled sample points for different classes.
2. The mouse position inside the canvas becomes the query point.
3. The program finds the `k` nearest labeled samples.
4. The majority class among those neighbors becomes the prediction.

In the visualization:

- Colored dots are labeled training samples.
- The glowing crosshair is the query point.
- Lines show which neighbors are currently being used for the decision.
- The predicted class updates live as you move the mouse.

Note: KNN is usually a classification or regression algorithm, not a clustering algorithm. It is included here because it is another beginner-friendly way to understand distance-based machine learning.

## Setup

Install Pygame if you do not already have it:

```bash
pip install pygame
```

## How To Run

Run the K-Means demo:

```bash
python clustering_example.py
```

Run the KNN demo:

```bash
python knn_example.py
```

## Controls

### K-Means Demo

- Left click inside the canvas: add a point
- `Start` button or `Space`: begin clustering
- `Scatter` button: generate random points
- `Reset` button or `R`: clear the canvas
- `[` or `-`: decrease `K`
- `]`, `=`, or `+`: increase `K`
- `Esc`: quit

### KNN Demo

- Left click inside the canvas: add a labeled sample
- `1`, `2`, `3`: switch the active class label
- Move the mouse in the canvas: test the query point
- `Scatter` button or `S`: load demo samples
- Right click near a point: remove the nearest sample
- `Reset` button or `R`: clear all samples
- `[` or `-`: decrease `k`
- `]`, `=`, or `+`: increase `k`
- `Esc`: quit

## Why This Project Is Useful

These examples are helpful for learning because they make the algorithms visible:

- You can see how distance affects grouping and prediction.
- You can experiment with different values of `K` or `k`.
- You can build intuition without needing a dataset file or a heavy ML library.

## Possible Next Steps

- Add DBSCAN for density-based clustering
- Add on-screen tutorial overlays
- Add dataset loading from CSV
- Add a decision boundary background for the KNN demo

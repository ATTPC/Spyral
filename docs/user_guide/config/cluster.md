# Clustering Configuration

The Cluster parameters which control the clustering, joining, and outlier detection algorithms. The default cluster parameters given in `config.json` are:

```json
"Cluster":
{
    "min_cloud_size": 50,
    "smoothing_neighbor_distance(mm)": 10.0,
    "minimum_points": 3,
    "big_event_cutoff": 300,
    "minimum_size_big_event": 50,
    "minimum_size_small_event": 10,
    "circle_overlap_ratio": 0.75,
    "fractional_charge_threshold": 0.75,
    "n_neighbors_outlier_test": 5
},
```

A break down of each parameter:

## min_cloud_size

This is the minimum size a point cloud must be (in number of points) to be sent through the clustering algorithm. Any smaller point clouds will be tossed as noise.

## smoothing_neighbor_distance(mm)

The neighborhood search radius in millimeters for the smoothing algorithm. Before data is clustered, it is smoothed by averaging over neighbors. This value should be small to avoid smoothing artifacts. This is a scale-dependent parameter and will need adjusted for each experiment.

## minimum_points

The minimum number of samples (points) in a neighborhood for a point to be a core point. This is a re-exposure of the `min_samples` parameter of [scikit-learn's HDBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html#sklearn.cluster.HDBSCAN). See their documentation for more details. This parameter needs more testing with more datasets! Please report any interesting behavior!

## big_event_cutoff

This is the size cutoff in points which identifies big events and small events. Big events have more points than `big_event_cutoff` and small events have fewer points than `big_event_cutoff`. This controls which minimum cluster size is applied to the event.

## minimum_size_big_event

The minimum number of samples (points) in a group for the group to be a valid cluster for big events (events with more points than `big_event_cutoff`). This is a re-exposure of the `min_cluster_size` parameter of [scikit-learn's HDBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html#sklearn.cluster.HDBSCAN). See their documentation for more details. This parameter needs more testing with more datasets! Please report any interesting behavior!

## minimum_size_small_event

The minimum number of samples (points) in a group for the group to be a valid cluster for small events (events with fewer points than `big_event_cutoff`). This is a re-exposure of the `min_cluster_size` parameter of [scikit-learn's HDBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html#sklearn.cluster.HDBSCAN). See their documentation for more details. This parameter needs more testing with more datasets! Please report any interesting behavior!

## circle_overlap_ratio

The minimum amount of overlap between circles fit to two clusters for the clusters to be joined together into a single cluster. HDBSCAN often fractures trajectories into multiple clusters as the point density changes due to the pad size, gaps, etc. These fragments are grouped together based on how much circles fit on their 2-D projections (X-Y) overlap.

## fractional_charge_threshold

The maximum allowed difference between the mean charge of two clusters (relative to the larger of the two means) for the clusters to be joined together into a single cluster. HDBSCAN often fractures trajectories into multiple clusters as the point density changes due to the pad size, gaps, etc. These fragments are grouped together based on how similar their mean charges are.

## n_neighbors_outlier_test

Number of neighbors to sample for the Local Outlier Test, used to remove outliers from clusters. This is a re-exposure of the `n_neighbors` parameter of [scikit-learn's LocalOutlierFactor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html). See their documentation for more details. In general smaller numbers results in more aggressive data cleaning.

# The Second Phase: ClusterPhase

Once point clouds have been created in the first phase, the next phase aims to break the point clouds into clusters.
Clusters are collections of points which are presumed to belong to a single particle trajectory. Clustering is a 
very common form of unsupervised machine learning, clustering takes a dataset and applies labels to samples (points)
based on the features (coordinates, charge). In this phase we make use of the 
[scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#clustering) algorithms.

There are three steps to this phase:

1. Initial clustering using HDBSCAN
2. Joining of clusters using circle-fitting and relative charge
3. Cleanup of joined clusters

Below we'll break down each of these steps.

## Initial Clustering with HDBSCAN

In some ways AT-TPC data doesn't fit very neatly into many of the clustering algorithms. The data is in general noisy,
trajectories are not isolated spatially, and the density of the points can vary within a single trajectory due to the
small/large pads on the pad plane. Additionally, there can be gaps in the trajectory due to the hole in the pad plane.
Thankfully, there is an algorithm, [HDBSCAN](https://scikit-learn.org/stable/modules/clustering.html#hdbscan) which
covers most of our bases.

HDBSCAN is a modified DBSCAN algorithm; DBSCAN stands for Density-Based Spatial Clustering of Applications wth Noise, and
is a powerful clustering algorithm for noisy data. It clusters points based on their density (closeness to neighbors), but
it assumes a *global* density. This means that DBSCAN assumes all clusters have the same density of points. As we
mentioned before, this is not so good for AT-TPC data which can change density quite a bit. HDBSCAN however does not have
this limitation. HDBSCAN scans density values forming clusters over different densities; the way it does this is fairly
involved and we direct you to the [scikit-learn docs](https://scikit-learn.org/stable/modules/clustering.html#hdbscan) for 
more details.

We cluster on the spatial coordinates (x, y, z). The z-axis is rescaled to make it the same total length as the x and y
coordinates (this helps to avoid giving to much weight to separation on the z-axis). HDBSCAN, as implemented in scikit-learn,
is also very fast, which is great because we have to do a lot of clustering!

However, this doesn't perfectly cluster our data right off the bat. HDBSCAN will make clusters over all the densities, but
each density will get it's own cluster! This is not exactly what we want; we need to connect these pieces together to form the
total trajectory.

Note that, before clustering, we prune the point cloud of any points which lie outside the legal bounds of the detector on the
z-axis (i.e. remove all points for which z < 0.0 or z > detector length). This helps us avoid including any extra noise in the
algorithm.

There are several parameters to HDBSCAN that are exposed in the [configuration](../config/cluster.md) of Spyral.

## Joining of Clusters

There are two available joining methods: the area overlap and continuity methods

### Area Overlap Method

The overlap method of joining is relatively simple: each initial cluster is fit with a circle in the X-Y plane. If the circles
overlap significantly they are taken to be from the same trajectory. The required amount of overlap is a 
[configuration](../config/cluster.md) parameter to be tuned for each experiment. Additionally, to avoid having noise clusters be
joined, a minimum size parameter is used such that clusters below the minimum size requirement are not evaluated in the joining. 
Joining is repeated until clusters can no longer be joined together. These clusters are then taken to represent the complete
detected particle trajectories for that event.

### Continuity Method

The continuity method aims to exploit the full 3-D information of the cluster rather than just the X-Y plane. The clusters are sorted
such that the largest, earliest clusters are compared first. Two clusters are joined if the end of each cluster which matches in time
(late end to early end) meet certain thresholds on the distance between the ends (see [configuration](../config/cluster.md) for details).
Joining is then repeated until clusters can no longer be joined together. These clusters are then taken to represent the complete
detected particle trajectories for that event.

### Which method should I use?

The overlap method is the traditional method, which has been used with success on several AT-TPC datasets. The continuity method was developed
to improve upon the overlap method. The continuity method tries to specifically address

- Events where there are multiple trajectories which overlap on the 2-D projection
- Very small clusters which get fragmented by HDBSCAN

The continuity method seems to improve these results. It has been tested to try and match the overlap method as much as possible on "normal"
events, however, the continuity method does struggle more with the "long proton" events. These are events which generally contain one very large
spiraling trajectory, which is fragmented into several clusters by HDBSCAN. Because the continuity method relies on time-ordering, if one of the
cluster segments is skipped, and others are joined around it, that skipped segment will be unable to be joined. This does not always happen, but
it can happen.

The general recommendation at the moment is that the overlap method should be used for publication-level analysis, while the continuity method
is considered more experimental and could be unreliable. This stance may change in the future as more testing is done.

## Cleanup of the Final Clusters

As a last stage, a final noise removal is applied to each cluster. The
[Local Outlier Factor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html#sklearn.neighbors.LocalOutlierFactor)
is an outlier test which compares the distance to nearest neighbors (neighbor density) to determine outlier points. The number of
neighbors is controlled using the [`outlier_scale_factor`](../config/cluster.md) parameter.

## Note

All of these steps will return a set of cluster representation (the acutal cluster objects) as well as an array of labels. The 
array of labels contains the label for each point in the *original* point cloud all of the clusters derived from. This is not really
used in the actual analysis, but it is extremely useful for plotting the effects of the clustering.

## Final Thoughts

The intial clustering and joining are two of the most important steps in Spyral, but they often require compromises. In our
experience, there is no perfect set of parameters that will allow the clustering to work flawlessly for all events in a dataset; in
fact given the inhomogenetiy of AT-TPC data, expecting one set of parameters to work all the time is somewhat crazy. In general, it
is best to shoot for parameters that *generally* work. There will always be events that are too noisy or too fragmented for the
clustering algorithm to work correctly. As such, this is the section of the code that most often requires fine tuning by hand.


# Clustering Configuration

The Cluster parameters which control the clustering, joining, and outlier detection algorithms.

The default recommended settings for HDBSCAN when using the continuity join method are:

```python
cluster_params = ClusterParameters(
    min_cloud_size=50,
    # hdbscan_parameters = None
    hdbscan_parameters = HdbscanParameters(
        min_points=5,
        min_size_scale_factor=0.0,
        min_size_lower_cutoff=5,
        cluster_selection_epsilon=13.0),
    overlap_join=None,
    continuity_join=ContinuityOverlapParamters(
        join_radius_fraction=0.3,
        join_z_fraction=0.2,
    ),
    # continuity_join = None
    outlier_scale_factor=0.05,
    direction_threshold = 0.5,
    tripclust_parameters = None,
)
```

The default recommended settings for HDBSCAN when using the circle overlap join method are:

```python
cluster_params = ClusterParameters(
    min_cloud_size=50,
    # hdbscan_parameters = None
    hdbscan_parameters = HdbscanParameters(
        min_points=3,
        min_size_scale_factor=0.05,
        min_size_lower_cutoff=10,
        cluster_selection_epsilon=10.0),
    # overlap_join=None,
    overlap_join=OverlapJoinParameters(
        min_cluster_size_join=15.0,
        circle_overlap_ratio=0.25,
    ),
    continuity_join=None,
    direction_threshold = 0.5,
    tripclust_parameters = None,
    outlier_scale_factor=0.05,
)
```

A break down of each parameter:

## min_cloud_size

This is the minimum size a point cloud must be (in number of points) to be sent through the clustering algorithm. Any smaller point clouds will be tossed as noise.

## minimum_points

The minimum number of samples (points) in a neighborhood for a point to be a core point. This is a re-exposure of the `min_samples` parameter of
[scikit-learn's HDBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html#sklearn.cluster.HDBSCAN). See
their documentation for more details. Larger values will make the algorithm more likely to identify points as noise. See the original
[HDBSCAN docs](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html#) for details on why this parameter is important and how it can impact the data.

## minimum_size_scale_factor

HDBSCAN requires a minimum size (the hyper parameter `min_cluster_size` in
[scikit-learn's HDBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html#sklearn.cluster.HDBSCAN)) in terms of samples for a group to
be considered a valid cluster. AT-TPC point clouds vary dramatically in size, from tens of points to thousands. To handle this wide scale, we use a scale factor
to determine the appropriate minimum size, where `min_cluster_size = minimum_size_scale_factor * n_cloud_points`. The default value was found through some testing,
and may need serious adjustment to produce best results. Note that the scale factor should be *small*. Update: in the case of continuity joining, scaling was not
needed.

## minimum_size_lower_cutoff

As discussed in the above `minimum_size_scale_factor`, we need to scale the `min_cluster_size` parameter to the size of the point cloud. However, there must be
a lower limit (i.e. you can't have a minimum cluster size of 0). This parameter sets the lower limit; that is any `min_cluster_size` calculated using the scale factor
that is smaller than this cutoff is replaced with the cutoff value. As an example, if the cutoff is set to 10 and the calculated value is 50, the calculated value would
be used. However, if the calculated value is 5, the cutoff would be used instead.

## cluster_selection_epsilon

A re-exposure of the `cluster_selection_epsilon` parameter of
[scikit-learn's HDBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html#sklearn.cluster.HDBSCAN). This parameter will merge clusters that
are less than epsilon apart. Note that this epsilon must be on the scale of the scaled data (i.e. it is not in normal units). The impact of this parameter is large, and
small changes to this value can produce dramatically different results. Larger values will bias the clustering to assume the point cloud is one single cluster (or all noise),
while smaller values will cause the algorithm to revert to the default result of HDBSCAN. See the original
[HDBSCAN docs](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html#) for details on why this parameter is important and how it can impact the data.

The recommended parameters when using the TRIPCLUST (Dalitz) clustering algorithm are:
(see also the publication [Dalitz clustering](https://doi.org/10.1016/j.cpc.2018.09.010) for more information)

```python
# tripclust_parameters=None,
tripclust_parameters=TripclustParameters(
    r=6,
    rdnn=True,
    k=12,
    n=3,
    a=0.03,
    s=0.3,
    sdnn=True,
    t=0.0,
    tauto=True,
    dmax=0.0,
    dmax_dnn=False,
    ordered=True,
    link=0,
    m=50,
    postprocess=False,
    min_depth=25,
```
Breakdown of each of the parameters:

## r

The neighbor distance for smoothing.

## rdnn

When this boolean is set to True the value of r is calculated automatically using dnn.

## k

The number of tested neighbors for each triplet mid point.

## n

The maximum number of triplets retained for each mid point.

## a

The maximum value of angle between the two triplet branches, expressed as 1-cos(alpha).

## s

The distance scale factor in the metric of triplet distance.

## sdnn

When set to True, the value of s is calculated automatically using dnn.

## t

The threshold for the "distance" between triplets.

## tauto

When set to True, the value of t is set automatically.

## dmax

The maximum gap width.

## dmax_dnn

Use dnn for dmax.

## ordered

When True the point cloud is ordered in chronological order.

## link

Linkage method for hierarchical clustering of the triplets.

## m

The minimum number of triplets per cluster.

## postprocess

When set to True, the post process algorithm is attempted.

## min_depth

Value of the minimum depth used in the post processing algorithm.



## outlier_scale_factor

We use [scikit-learn's LocalOutlierFactor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html) as a last round of noise elimination on
a cluster-by-cluster basis. This algorithm requires a number of neighbors to search over (the `n_neighbors` parameter). As with the `min_cluster_size` in HDBSCAN, we need to
scale this value off the size of the cluster. This factor multiplied by the size of the cluster gives the number of neighbors to search over
(`n_neighbors = outlier_scale_factor * cluster_size`). This value tends to have a "sweet spot" where it is most effective. If it is too large, every point has basically the
same outlier factor as you're including the entire cluster for every point. If it is too small the variance between neighbors can be too large and the results will be
unpredictable. Note that if the value of `outlier_scale_factor * cluster_size` is less than 2, `n_neighbors` will be set to 2 as this is the minimum allowed value.
This algorithm is also used to clean clusters prior to the joining process, using the same parameter.

## Overlap Join Parameters

### min_cluster_size_join

The minimum size of a cluster for it to be considered in the joining step of the clustering. After HDBSCAN has made the initial clusters we attempt to combine any clusters which
have overlapping circles in the 2-D projection (see `circle_overlap_ratio`). However, many times, small pockets of noise will be clustered and often sit within the larger trajectory.
To avoid these being joined we require a cluster to have a minimum size.

### circle_overlap_ratio

The minimum amount of overlap between circles fit to two clusters for the clusters to be joined together into a single cluster. HDBSCAN often fractures trajectories into multiple
clusters as the point density changes due to the pad size, gaps, etc. These fragments are grouped together based on how much circles fit on their 2-D projections (X-Y) overlap.

## Continuity Join Parameters

### join_radius_fraction

The fraction of the total radius range of the two clusters used to set the threshold for joining. Used as: `radius_threshold = join_radius_fraction * (max_radius - min_radius)`
where the `min_radius` and `max_radius` are the minimum and maximum radius values over the clusters being compared.

### join_z_fraction

The fraction of the total z range of the two clusters used to set the threshold for joining. Used as: `z_threshold = join_z_fraction * (max_z - min_z)`
where the `min_z` and `max_z` are the minimum and maximum radius values over the clusters being compared.

# **Chapter-3: Quizes**

---

!!! note "Quiz"
    **1. What is the primary motivation for using dimensionality reduction, as described by the Manifold Hypothesis?**

    - A. To increase the computational speed of matrix multiplication.
    - B. To add noise to the data for better generalization.
    - C. The belief that high-dimensional physical data is confined to a much lower-dimensional manifold, making the raw data intractably sparse.
    - D. To ensure all features have a Gaussian distribution.

    ??? info "See Answer"
        **Correct: C**

        *(The Curse of Dimensionality makes high-D spaces mostly empty. The Manifold Hypothesis states that physical constraints confine data to a low-D surface, which DR aims to find.)*

---

!!! note "Quiz"
    **2. In the context of physics, dimensionality reduction is the computational analog of what theoretical process?**

    - A. Calculating a partition function.
    - B. Coarse-graining, where microscopic degrees of freedom are projected onto a few collective coordinates or order parameters.
    - C. Solving Schrödinger's equation.
    - D. Performing a Legendre transform.

    ??? info "See Answer"
        **Correct: B**

        *(DR automates the discovery of collective variables (like hydrodynamic fields from molecular trajectories) directly from data, a task that traditionally requires deep theoretical insight.)*

---

!!! note "Quiz"
    **3. What is the core mathematical operation at the heart of Principal Component Analysis (PCA)?**

    - A. Fourier transformation of the time-series data.
    - B. Minimizing the KL divergence between two distributions.
    - C. Eigendecomposition of the data's covariance matrix ($\Sigma$).
    - D. Calculating the geodesic distance between all pairs of points.

    ??? info "See Answer"
        **Correct: C**

        *(PCA finds the eigenvectors ($\mathbf{v}_k$) and eigenvalues ($\lambda_k$) of the covariance matrix, where the eigenvectors are the principal axes of the data cloud.)*

---

!!! note "Quiz"
    **4. In PCA, what is the physical interpretation of the first principal component, $\mathbf{v}_1$?**

    - A. The average state of the system.
    - B. A random direction in the feature space.
    - C. The direction of minimum variance in the data.
    - D. The system's dominant collective mode of motion, representing the direction of greatest variance.

    ??? info "See Answer"
        **Correct: D**

        *(For example, in molecular dynamics, $\mathbf{v}_1$ often corresponds to a large-scale, coordinated motion like the hinging of two protein domains.)*

---

!!! note "Quiz"
    **5. How is the intrinsic dimensionality of a dataset typically estimated using PCA?**

    - A. By counting the number of features with non-zero mean.
    - B. By finding the number of principal components $d$ required to capture a large fraction (e.g., 95%) of the total variance.
    - C. By taking the logarithm of the number of samples.
    - D. By the number of clusters found by K-Means.

    ??? info "See Answer"
        **Correct: B**

        *(This is done by plotting the cumulative explained variance, $\sum \lambda_k / \sum \lambda_j$, and finding the "knee" of the curve.)*

---

!!! note "Quiz"
    **6. What is the fundamental limitation of PCA that necessitates the use of nonlinear methods like t-SNE and UMAP?**

    - A. It is too computationally expensive for large datasets.
    - B. It assumes the underlying data manifold is linear (a flat hyperplane) and fails on curved or "Swiss-rolled" data.
    - C. It can only be applied to 2D data.
    - D. It requires the data to be noise-free.

    ??? info "See Answer"
        **Correct: B**

        *(PCA will distort curved manifolds by projecting them onto a flat plane, often folding distinct states on top of each other.)*

---

!!! note "Quiz"
    **7. What is the guiding principle of manifold learning algorithms like t-SNE and UMAP?**

    - A. To preserve the global variance of the data cloud.
    - B. To ensure the low-dimensional embedding is perfectly uncorrelated.
    - C. To preserve the local geometry and neighborhood relationships of the data.
    - D. To maximize the distance between all points in the embedding.

    ??? info "See Answer"
        **Correct: C**

        *(The core idea is that points that are neighbors on the curved manifold should remain neighbors in the low-dimensional map.)*

---

!!! note "Quiz"
    **8. t-SNE works by minimizing the KL divergence between two probability distributions. What do these distributions, $P_{ij}$ and $Q_{ij}$, represent?**

    - A. $P_{ij}$ is the prior and $Q_{ij}$ is the posterior.
    - B. $P_{ij}$ represents neighborhood probabilities in the high-D space, and $Q_{ij}$ represents neighborhood probabilities in the low-D map.
    - C. $P_{ij}$ is for training data and $Q_{ij}$ is for test data.
    - D. $P_{ij}$ is the energy and $Q_{ij}$ is the entropy.

    ??? info "See Answer"
        **Correct: B**

        *(t-SNE tries to make the low-D map's neighborhood structure match the original high-D structure by minimizing the "distance" between these two distributions.)*

---

!!! note "Quiz"
    **9. When interpreting a t-SNE or UMAP plot, what is a common pitfall to avoid?**

    - A. Assuming the axes correspond to principal components.
    - B. Interpreting the global distances between clusters and the relative sizes of clusters as being physically meaningful.
    - C. Believing that the plot has preserved any information from the high-D space.
    - D. Thinking that clusters in the plot correspond to metastable states.

    ??? info "See Answer"
        **Correct: B**

        *(These algorithms preserve local topology at the expense of global geometry. The size and separation of clusters are often artifacts and should not be over-interpreted.)*

---

!!! note "Quiz"
    **10. In the context of a physical simulation, what does a dense cluster of points in a data cloud correspond to?**

    - A. A high-energy transition state.
    - B. A region of phase space that was not sampled.
    - C. A low-energy basin of attraction, or a metastable state.
    - D. An artifact of the integration algorithm.

    ??? info "See Answer"
        **Correct: C**

        *(The system spends most of its time in low-energy states, leading to a high density of samples in those regions of the data manifold.)*

---

!!! note "Quiz"
    **11. What is the objective function that the K-Means algorithm seeks to minimize?**

    - A. The total number of clusters.
    - B. The distance between the two farthest clusters.
    - C. The intra-cluster variance (sum of squared Euclidean distances from each point to its assigned centroid).
    - D. The entropy of the cluster assignments.

    ??? info "See Answer"
        **Correct: C**

        *(The cost function is $J = \sum_{ik} r_{ik} \|\mathbf{x}_i - \mathbf{\mu}_k\|^2$. Minimizing it makes the clusters as "tight" as possible.)*

---

!!! note "Quiz"
    **12. The iterative two-step process used by K-Means (Lloyd's algorithm) consists of which two steps?**

    - A. E-Step: Update centroids; M-Step: Assign points.
    - B. E-Step: Assign points to nearest centroid; M-Step: Update centroids to the mean of assigned points.
    - C. E-Step: Calculate variance; M-Step: Calculate mean.
    - D. E-Step: Project data; M-Step: Cluster data.

    ??? info "See Answer"
        **Correct: B**

        *(This is a classic example of a coordinate descent or relaxation algorithm that is guaranteed to find a local minimum of the objective function.)*

---

!!! note "Quiz"
    **13. What is a primary advantage of a density-based clustering algorithm like DBSCAN over a centroid-based one like K-Means?**

    - A. It is much faster.
    - B. It can find clusters of arbitrary, non-convex shapes and is robust to outliers.
    - C. It does not require any input parameters.
    - D. It always finds the globally optimal number of clusters.

    ??? info "See Answer"
        **Correct: B**

        *(DBSCAN defines clusters as continuous regions of high density, allowing it to identify "S"-shaped or nested clusters where K-Means would fail.)*

---

!!! note "Quiz"
    **14. In DBSCAN, what is a "core point"?**

    - A. Any point that belongs to a cluster.
    - B. The geometric center of a cluster.
    - C. A point that has at least `minPts` neighbors within a radius of `epsilon`.
    - D. A point that is labeled as noise.

    ??? info "See Answer"
        **Correct: C**

        *(Core points are the seeds from which dense clusters are grown. Points that are not core points but are neighbors of one are "border points".)*

---

!!! note "Quiz"
    **15. What is the output of a hierarchical clustering algorithm?**

    - A. A set of K cluster centroids.
    - B. A label for each point indicating noise or cluster membership.
    - C. A tree-like structure called a dendrogram that shows the hierarchy of cluster merges.
    - D. A low-dimensional embedding of the data.

    ??? info "See Answer"
        **Correct: C**

        *(The dendrogram can then be "cut" at a certain height to produce a flat clustering with a desired number of clusters.)*

---

!!! note "Quiz"
    **16. A Gaussian Mixture Model (GMM) performs "soft clustering." What does this mean?**

    - A. The cluster boundaries are not well-defined.
    - B. It assigns each data point a probability of belonging to each of the K clusters.
    - C. The algorithm is not guaranteed to converge.
    - D. It can only be used for very small datasets.

    ??? info "See Answer"
        **Correct: B**

        *(Instead of a hard assignment (point X is in cluster 2), GMM provides a posterior probability distribution, e.g., P(cluster=1|X)=0.1, P(cluster=2|X)=0.9.)*

---

!!! note "Quiz"
    **17. The Expectation-Maximization (EM) algorithm used to train GMMs is analogous to what physical process?**

    - A. A system relaxing to a state of minimum free energy, $\mathcal{F} = \langle E \rangle - TS$.
    - B. A particle undergoing Brownian motion.
    - C. A quantum system tunneling through a barrier.
    - D. A system evolving under Liouville's theorem.

    ??? info "See Answer"
        **Correct: A**

        *(The EM algorithm can be formally shown to be a process that minimizes a free energy functional, where the E-step updates probabilities and the M-step updates parameters to lower the energy.)*

---

!!! note "Quiz"
    **18. What is the recommended two-step workflow for discovering phases in a high-dimensional simulation dataset with complex geometry?**

    - A. PCA, followed by K-Means.
    - B. UMAP (or t-SNE), followed by DBSCAN.
    - C. K-Means, followed by PCA.
    - D. DBSCAN, followed by UMAP.

    ??? info "See Answer"
        **Correct: B**

        *(First, use a nonlinear method like UMAP to create a curvature-aware 2D map. Second, use a density-based method like DBSCAN on that map to identify the arbitrary-shaped clusters (phases).)*

---

!!! note "Quiz"
    **19. In hierarchical clustering, what does the "linkage criterion" define?**

    - A. The number of clusters to find.
    - B. How to measure the distance between two clusters (not just two points).
    - C. The final shape of the dendrogram.
    - D. The initialization method for the clusters.

    ??? info "See Answer"
        **Correct: B**

        *(Examples include 'single' (closest points), 'complete' (farthest points), and 'average' linkage, each of which results in different cluster shapes.)*

---

!!! note "Quiz"
    **20. Why is the geodesic distance considered more physically meaningful than Euclidean distance on a curved manifold?**

    - A. It is faster to compute.
    - B. It measures the straight-line path through the high-dimensional embedding space.
    - C. It measures the shortest path *along the curved surface* of the manifold, representing the true path a system must take.
    - D. It is always a smaller value than the Euclidean distance.

    ??? info "See Answer"
        **Correct: C**

        *(The Euclidean distance can "cheat" by cutting through high-energy barriers, while the geodesic distance correctly captures the length of the low-energy path.)*

---

!!! note "Quiz"
    **21. The K-Means algorithm is guaranteed to converge to:**

    - A. The globally optimal clustering.
    - B. A local minimum of the intra-cluster variance objective function.
    - C. Exactly K clusters of equal size.
    - D. A state where all clusters are perfectly spherical.

    ??? info "See Answer"
        **Correct: B**

        *(Because it is a greedy descent algorithm, its final solution depends on the initial random placement of centroids. It is not guaranteed to find the global optimum.)*

---

!!! note "Quiz"
    **22. In the provided code example for quantifying dimensionality, why is the data standardized before applying PCA?**

    - A. To increase the number of dimensions.
    - B. To ensure all features are on a comparable scale (mean=0, variance=1), preventing features with large values from dominating the covariance calculation.
    - C. To make the data nonlinear.
    - D. To reduce the number of samples.

    ??? info "See Answer"
        **Correct: B**

        *(PCA is sensitive to the scale of the features. Standardization is a crucial preprocessing step to ensure the discovered principal components reflect the correlation structure, not arbitrary units.)*

---

!!! note "Quiz"
    **23. The code demo comparing PCA and t-SNE on an S-curve shows that PCA "folds" the curve. This is a failure to preserve what property?**

    - A. The total variance of the data.
    - B. The mean of the data.
    - C. The local neighborhood structure and topology of the manifold.
    - D. The number of data points.

    ??? info "See Answer"
        **Correct: C**

        *(Points that are far apart on the "unrolled" S-curve are projected close together by PCA, destroying the true topological relationship that t-SNE successfully preserves.)*

---

!!! note "Quiz"
    **24. In the "Automated Phase Discovery" code example, why is DBSCAN a better choice than K-Means for clustering the `make_moons` dataset?**

    - A. The dataset has exactly two clusters.
    - B. The dataset is perfectly linear.
    - C. The clusters ("moons") are non-convex (crescent-shaped), a geometry that K-Means' spherical assumption cannot handle.
    - D. The dataset contains no noise.

    ??? info "See Answer"
        **Correct: C**

        *(K-Means would try to partition the crescents with a straight line, incorrectly splitting them. DBSCAN's density-based approach correctly identifies each moon as a single, continuous cluster.)*

---

!!! note "Quiz"
    **25. The convergence plot for the K-Means algorithm shows that the objective function J:**

    - A. Increases at every iteration.
    - B. Fluctuates randomly before converging.
    - C. Monotonically decreases until it reaches a stable local minimum.
    - D. Stays constant throughout the algorithm.

    ??? info "See Answer"
        **Correct: C**

        *(Each step of Lloyd's algorithm (assignment and update) is guaranteed to decrease or maintain the cost J, demonstrating it is a relaxation process into a local energy well.)*


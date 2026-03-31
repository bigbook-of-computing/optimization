# **Chapter 3: Dimensionality Reduction & Clustering**

# **Introduction**

Chapters 1 and 2 revealed a fundamental paradox: simulation data naturally lives in extremely high-dimensional spaces $\mathbb{R}^D$ (where $D$ might be thousands or millions), yet the **Curse of Dimensionality** renders all naive statistical methods—density estimation, nearest-neighbor search, visualization—completely ineffective in these spaces. The resolution to this paradox lies in the **Manifold Hypothesis**: real physical data, despite being embedded in $\mathbb{R}^D$, is not uniformly distributed throughout this vast space. Instead, it is confined to a much lower-dimensional manifold $\mathcal{M}$ of intrinsic dimension $d \ll D$, constrained by conservation laws, physical interactions, and the system's governing equations. A protein trajectory explores only a tiny subspace of all possible atomic configurations; a gas of $N$ particles is constrained by energy and momentum conservation; a phase transition concentrates probability mass along a narrow, one-dimensional order parameter.

This chapter provides the algorithmic toolkit for **discovering and extracting this hidden low-dimensional structure**—a process known as **dimensionality reduction**. We begin by establishing the physical and mathematical motivation: dimensionality reduction is the computational analog of **coarse-graining** in physics, the process of projecting microscopic degrees of freedom onto a small set of collective coordinates or order parameters. We develop **Principal Component Analysis (PCA)**, the foundational linear method that identifies the axes of maximum variance via eigendecomposition of the covariance matrix—mathematically equivalent to discovering the dominant normal modes of a fluctuating system. We then confront PCA's critical limitation (it assumes flat geometry) and introduce **nonlinear manifold learning methods**—**t-SNE** and **UMAP**—which preserve local neighborhood topology and can "unroll" curved manifolds into interpretable 2D or 3D maps. Having compressed the data into a tractable low-dimensional representation, we then turn to **clustering**, the algorithmic task of partitioning this landscape into distinct groups or phases. We develop **K-Means** (a centroid-based energy minimizer), **DBSCAN** (a density-based method robust to arbitrary cluster shapes), and **Gaussian Mixture Models** (a probabilistic approach equivalent to free-energy minimization via Expectation-Maximization).

By the end of this chapter, you will understand how to apply the complete data-driven pipeline: take raw high-dimensional trajectories, apply PCA or UMAP to project them into a 2D or 3D latent space, and then use clustering to automatically identify the system's metastable states or phases—all without prior knowledge of the governing physics. You will recognize that the discovered clusters correspond to basins of attraction in the free-energy landscape, and that this entire workflow is the computational realization of the physicist's goal of finding order parameters. These techniques prepare you for **Chapter 4**, where we shift from analyzing static data landscapes to actively **navigating and optimizing** loss landscapes—the core task of machine learning and computational optimization.

---

# **Chapter 3: Outline**

| **Sec.** | **Title**                                                 | **Core Ideas & Examples**                                                                                                                                                                                      |
| -------- | --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **3.1**  | **Why Reduce Dimensionality?**                            | Intrinsic dimension $d \ll$ embedding dimension $D$; Manifold Hypothesis; physics analogy to coarse-graining (hydrodynamic fields from molecular trajectories); order parameters as low-dimensional projections; mathematical goal $f: \mathbb{R}^D \to \mathbb{R}^d$. |
| **3.2**  | **Linear Methods: Principal Component Analysis (PCA)**    | Covariance eigendecomposition $\Sigma \mathbf{v}_k = \lambda_k \mathbf{v}_k$; principal components as collective modes; projection $\mathbf{z}_i = V_d^\top (\mathbf{x}_i - \mathbf{\mu})$; explained variance ratio $\lambda_k / \sum \lambda_j$; connection to normal mode analysis in molecular dynamics. |
| **3.3**  | **Beyond Linear Geometry: Nonlinear Embeddings**          | Limitations of PCA on curved manifolds (Swiss roll, pendulum state space); manifold learning preserves local neighborhoods; **t-SNE**: high-D Gaussian probabilities $P_{ij}$, low-D Student-t probabilities $Q_{ij}$, minimize KL divergence $D_{\mathrm{KL}}(P||Q)$; **UMAP**: fuzzy simplicial sets, topological structure preservation. |
| **3.4**  | **Metrics and Similarity Preservation**                   | Global vs. local metrics (Euclidean, Mahalanobis, cosine similarity); PCA preserves global variance, t-SNE/UMAP preserve local topology; curvature awareness and distortion; evaluating embedding quality via pairwise distance matrix comparison $D_{\text{high}}$ vs $D_{\text{low}}$. |
| **3.5**  | **Clustering: Discovering Groups and Phases**             | Partition data into $K$ clusters $\{C_1, \dots, C_K\}$; physical metaphor—clusters as basins of attraction in free-energy landscape; taxonomy: centroid-based (K-Means), density-based (DBSCAN), connectivity-based (hierarchical), probabilistic (GMM). |
| **3.6**  | **K-Means: The Simplest Energy Minimizer**                | Objective function (intra-cluster variance) $J = \sum_{ik} r_{ik} \|\mathbf{x}_i - \mathbf{\mu}_k\|^2$; Lloyd's algorithm (E-step: assign to nearest centroid, M-step: update centroids to cluster means); harmonic potential analogy; limitations: assumes convex/isotropic clusters, initialization sensitivity (k-means++). |
| **3.7**  | **Hierarchical and Density-Based Methods**                | **Hierarchical clustering**: agglomerative (bottom-up), dendrogram tree, linkage criteria (single/complete/average); **DBSCAN**: $\epsilon$-neighborhood, $minPts$, core/border/noise points; identifies arbitrary-shaped clusters; robust to outliers; workflow: UMAP → DBSCAN, overlay cluster labels on 2D map. |
| **3.8**  | **Probabilistic and Energy-Based Clustering**             | **Gaussian Mixture Model (GMM)** $p(\mathbf{x}) = \sum_k \pi_k \mathcal{N}(\mathbf{x}|\mathbf{\mu}_k, \Sigma_k)$; soft assignments via posterior $p(k|\mathbf{x}_i)$; **Expectation-Maximization (EM)** algorithm (E-step: compute responsibilities, M-step: update parameters); EM as free-energy minimization $\mathcal{F} = \langle E \rangle - TS$. |
| **3.9**  | **Worked Example: Discovering Phases in Simulation Data** | 2D Ising model at varying temperatures $T$; flatten $L \times L$ spin configurations to $\mathbb{R}^{L^2}$ vectors; apply UMAP for nonlinear projection to $\mathbb{R}^2$; DBSCAN clustering reveals three phases: ferromagnetic up ($M \approx +1$), ferromagnetic down ($M \approx -1$), paramagnetic disordered ($M \approx 0$); automatic phase discovery without prior knowledge. |
| **3.10** | **Code Demo: PCA + Clustering**                           | Generate synthetic 5D data with $K=3$ clusters; PCA projects to 2D principal subspace; K-Means partitions 2D map; visualize PC1 vs PC2 scatter plot colored by discovered cluster labels; demonstrates `PCA → K-Means` pipeline for phase/state identification from high-dimensional unlabeled data. |
| **3.11** | **Takeaways & Bridge to Chapter 4**                       | Chapter 3 workflow: UMAP/PCA finds manifold coordinates, clustering finds metastable states/phases; shift from **observer** (analyzing static datasets) to **agent** (navigating loss landscapes); clusters as basins we *found* → minima as basins we must *seek*; bridge to optimization: machine learning as physical dynamics on high-dimensional loss surfaces. |

---

## **3.1 Why Reduce Dimensionality?**

---

### **The Necessity of Compression**

In Chapters 1 and 2, we confronted a fundamental tension. We can represent physical simulations as data clouds in $\mathbb{R}^D$, but owing to the **Curse of Dimensionality**, these clouds are intractably sparse. We cannot effectively compute densities, find nearest neighbors, or visualize the landscape in its native high-dimensional space.

Yet, we know that physical systems are not arbitrary. A gas of $N$ particles does not explore all $3N$ spatial dimensions randomly; it is constrained by conservation laws (energy, momentum). A protein does not explore all possible atomic configurations; it is constrained by covalent bonds and steric clashes.

These constraints mean that the **intrinsic dimension** $d$ of the data manifold $\mathcal{M}$ is often orders of magnitude smaller than the **embedding dimension** $D$.

$$
d \ll D
$$

**Dimensionality reduction** is the mathematical formalism for finding this intrinsic lower-dimensional space. It is not merely a convenience for visualization; it is a necessary preprocessing step to make high-dimensional statistical inference possible. By compressing data from $\mathbb{R}^D$ to $\mathbb{R}^d$, we increase its effective density, allowing algorithms to find patterns that were previously lost in the vast emptiness of the ambient space.

---

### **Physics Analogy: Coarse-Graining and Order Parameters**

For the physicist, dimensionality reduction is a familiar concept under a different name: **coarse-graining**.

Consider a fluid. Microscopically, it is described by the positions and momenta of $10^{23}$ molecules ($D \approx 6 \times 10^{23}$). Macroscopically, however, we describe it perfectly well with just a few field variables: density $\rho(\mathbf{r})$, velocity $\mathbf{v}(\mathbf{r})$, and temperature $T(\mathbf{r})$. These hydrodynamic variables are *low-dimensional projections* of the microscopic state. They work because they capture the slow, collective modes of the system, while ignoring the fast, irrelevant microscopic fluctuations.

Similarly, in the theory of phase transitions, we distill complex microstates into a single **order parameter** (like magnetization $M$) that captures the essential physics of the transition.

In standard physics, finding these low-dimensional variables requires deep theoretical insight. In data science, our goal is to **automate this discovery**. We seek algorithms that can look at raw simulation data and *learn* the optimal coarse-grained variables (latent variables) without prior knowledge of the governing equations.

!!! tip "Dimensionality Reduction as Automated Order Parameter Discovery"
    The physicist manually chooses order parameters (magnetization, density, temperature) based on theoretical understanding. Machine learning algorithms like PCA and UMAP automate this process—they discover the collective coordinates that best describe the system's variability directly from data, without requiring prior physical insight.
    
---

### **The Mathematical Goal**

Formally, given a dataset $X = \{\mathbf{x}_1, \dots, \mathbf{x}_N\}$ where $\mathbf{x}_i \in \mathbb{R}^D$, we seek a mapping function $f$:

$$
f: \mathbb{R}^D \to \mathbb{R}^d
$$

that maps each high-dimensional observation to a low-dimensional vector $\mathbf{z}_i = f(\mathbf{x}_i)$, often called the **latent representation** or **embedding**.

Crucially, this mapping must preserve some essential structure of the original data. Different algorithms prioritize different structures:
* **Variance (PCA):** Preserves the "spread" of the data cloud.
* **Topology (t-SNE/UMAP):** Preserves local neighborhood relationships (keeping similar points close).
* **Distances (MDS/Isomap):** Preserves global geodesic distances across the manifold.

Choosing the right dimensionality reduction technique is equivalent to choosing *which* physical features we believe are most important to retain in our simplified model.

---

## **3.2 Linear Methods: Principal Component Analysis (PCA)**

**Principal Component Analysis (PCA)** is the foundational algorithm for linear dimensionality reduction. It provides the optimal *linear* mapping (in the least-squares sense) to a lower-dimensional subspace by finding the directions of maximum variance in the data. As we saw in Chapter 1, this is achieved by performing an eigendecomposition of the data's covariance matrix.

---

### **The Covariance Eigen-Decomposition**

Recall from Section 1.2 the $D \times D$ sample covariance matrix $\Sigma$:

$$
\Sigma = \frac{1}{N-1}\sum_{i=1}^N (\mathbf{x}_i - \mathbf{\mu})(\mathbf{x}_i - \mathbf{\mu})^\top
$$

PCA is simply the process of finding the eigenvectors $\mathbf{v}_k$ and eigenvalues $\lambda_k$ of this matrix. The eigenvectors (called **Principal Components**) define the axes of a new, data-aligned coordinate system, and the eigenvalues quantify the variance along those axes. They are found by solving the eigenvalue equation:

$$
\Sigma \mathbf{v}_k = \lambda_k \mathbf{v}_k, \quad k = 1, \dots, D
$$By convention, the eigenvectors are sorted in descending order of their corresponding eigenvalues, such that $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_D \ge 0$.

  * **PC1 ($\mathbf{v}_1$):** The first principal component is the $D$-dimensional vector that points in the direction of the *greatest* variance in the data.
  * **PC2 ($\mathbf{v}_2$):** The second principal component is the direction of greatest variance that is *orthogonal* to $\mathbf{v}_1$.

---

### **Projection onto the Principal Subspace**

Dimensionality reduction is achieved by truncating this new coordinate system. We *choose* to keep only the first $d$ components (where $d \ll D$), which form the **principal subspace**.

The projection of a single high-dimensional data point $\mathbf{x}_i$ onto this subspace is computed by:

1.  **Centering** the data point: $(\mathbf{x}_i - \mathbf{\mu})$
2.  **Projecting** it onto each of the $d$ principal components.

This is most efficiently written as a matrix multiplication. We define $V_d$ as a $D \times d$ matrix whose columns are the first $d$ eigenvectors $\{\mathbf{v}_1, \dots, \mathbf{v}_d\}$. The new $d$-dimensional representation $\mathbf{z}_i$ for data point $\mathbf{x}_i$ is:

$$
\mathbf{z}_i = V_d^\top (\mathbf{x}_i - \mathbf{\mu})
$$

The vector $\mathbf{z}_i \in \mathbb{R}^d$ is the "shadow" or latent representation of the original $\mathbf{x}_i \in \mathbb{R}^D$.

---

### **Interpretation: Collective Modes and Explained Variance**

The power of PCA is in its interpretation. The principal components $\mathbf{v}_k$ are almost never simple, individual features. They are **collective coordinates**—weighted linear combinations of *all* original $D$ features.

* **Physical Analogy (Molecular Dynamics):** This is the most direct physical parallel. In analyzing a molecular dynamics trajectory, PCA is mathematically equivalent to a **normal mode analysis** (or "essential dynamics") on the atomic fluctuations. The first principal component $\mathbf{v}_1$ is not "atom 3 moved left"; it is a complex vector describing a *collective motion* like the "hinging" of two protein domains, a motion that involves thousands of atoms moving in a correlated fashion [1]. PCA, therefore, automatically discovers the dominant, low-frequency normal modes of the system from the raw trajectory data.

* **Explained Variance Ratio:** How do we choose $d$? We use the eigenvalues. The **explained variance ratio** of component $k$ is the fraction of the total variance captured by that component:

$$
\text{EVR}_k = \frac{\lambda_k}{\sum_{j=1}^D \lambda_j}
$$

We choose $d$ by examining the **cumulative explained variance** $\sum_{k=1}^d \text{EVR}_k$. A common heuristic is to choose $d$ large enough to capture, for example, 90% of the total variance. In many physical systems, we find that the first $d=2$ or $d=3$ components capture a vast majority of the variance, confirming that the system's dynamics are confined to a very low-dimensional manifold.

---

## **3.3 Beyond Linear Geometry — Nonlinear Embeddings**

---

### **The Limitation of PCA: The "Curved Manifold" Problem**

Principal Component Analysis (PCA) is powerful, but it has a fundamental limitation: it is a **linear** projection. PCA describes the data cloud using a flat, $d$-dimensional hyperplane (a "hyper-pancake"). It assumes that the important directions of variation are straight lines.

This assumption fails for many physical systems. Consider a simple pendulum: its state space $(\theta, \dot{\theta})$ is curved. Or consider a molecule rotating in 3D: the manifold of its possible configurations is a **curved manifold**, not a flat plane. If we flatten these states into $\mathbb{R}^D$, the resulting data cloud will be twisted, "S"-shaped, or "Swiss-rolled" inside the embedding space.

PCA will fail dramatically on such data. It will attempt to "fit" a flat plane through the curved structure, leading to a projection that distorts the data, folds distinct states on top of each other, and completely misrepresents the system's true low-dimensional geometry.

!!! example "The Swiss Roll Problem"
    Imagine a 2D ribbon rolled into a 3D spiral (the "Swiss roll" dataset). The intrinsic dimensionality is $d=2$ (two coordinates describe position on the ribbon), but it's embedded in $\mathbb{R}^3$. PCA will project this onto a 2D plane, but the projection will overlap and fold distant parts of the ribbon onto each other. Nonlinear methods like UMAP successfully "unroll" the spiral back into a flat 2D rectangle, preserving the neighborhood structure.
    
---

### **Manifold Learning: Preserving Local Neighborhoods**

To solve this, we need **nonlinear dimensionality reduction**, also known as **manifold learning**. The guiding principle of these methods is to abandon the goal of preserving *global* linear variance (like PCA) and instead focus on preserving the **local geometry** of the data.

The core intuition is this:
> Two data points that are "neighbors" (close to each other) on the curved manifold $\mathcal{M}$ should also be neighbors in the final 2D or 3D visualization.

This approach seeks to "unroll" the manifold, much like peeling an orange and laying the peel flat on a table, by carefully preserving the local relationships between nearby points.

---

### **t-Distributed Stochastic Neighbor Embedding (t-SNE)**

**t-SNE** is perhaps the most popular nonlinear method for visualization [2]. It is a probabilistic method that operates in two steps:

1.  **High-Dimensional Probabilities ($P_{ij}$):** First, t-SNE builds a probability distribution in the high-D space $\mathbb{R}^D$. It models the probability $p_{j|i}$ that point $\mathbf{x}_i$ would pick $\mathbf{x}_j$ as its neighbor. This probability is modeled as a Gaussian centered at $\mathbf{x}_i$:

$$
p_{j|i} = \frac{\exp(-\|\mathbf{x}_i - \mathbf{x}_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|\mathbf{x}_i - \mathbf{x}_k\|^2 / 2\sigma_i^2)}
$$

```
(The bandwidth $\sigma_i$ is adaptively chosen for each point.) The final probability $P_{ij}$ is a symmetrized version, $P_{ij} = (p_{j|i} + p_{i|j}) / 2N$.

```
2.  **Low-Dimensional Probabilities ($Q_{ij}$):** Second, t-SNE creates a similar probability distribution for the points $\mathbf{z}_i$ in the low-D "map" space (e.g., $\mathbb{R}^2$). Critically, it uses a **Student's t-distribution** (which has "heavy tails") instead of a Gaussian:

$$
Q_{ij} = \frac{(1 + \|\mathbf{z}_i - \mathbf{z}_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|\mathbf{z}_k - \mathbf{z}_l\|^2)^{-1}}
$$

3.  **Minimizing KL Divergence:** t-SNE's objective is to find the set of points $\{\mathbf{z}_i\}$ in the low-D map that minimizes the **Kullback-Leibler (KL) divergence** (Section 2.2) between the two probability distributions, $P$ and $Q$:

$$
\text{Cost} = D_{\mathrm{KL}}(P||Q) = \sum_{i \neq j} P_{ij} \ln \frac{P_{ij}}{Q_{ij}}
$$

This minimization acts like a complex force-directed graph layout. It "pulls" points together in the 2D map if they were neighbors in the high-D space ($P_{ij}$ is high) and "pushes" them apart if they were distant ($P_{ij}$ is low).

---

### **Uniform Manifold Approximation and Projection (UMAP)**

**UMAP** is a more recent technique that has gained enormous popularity, often producing clearer visualizations with faster computation times [3]. While its mathematical derivation is more complex (rooted in topological data analysis and fuzzy simplicial sets), its core intuition is similar to t-SNE:
1.  It constructs a **graph** of local neighborhood relationships in the high-D space.
2.  It then seeks a low-D embedding that preserves the **topological structure** of this high-D graph.

In practice, both t-SNE and UMAP are powerful "black-box" tools for taking a high-D data cloud and producing a 2D or 3D scatter plot that reveals its underlying cluster and manifold structure, far more faithfully than a linear PCA projection could.

---

## **3.4 Metrics and Similarity Preservation**

The choice of a dimensionality reduction algorithm (linear PCA or nonlinear UMAP) is a choice about *what* geometric properties to preserve. This, in turn, is governed by the underlying **metric**, or distance function, used to measure similarity between points.

---

### **Global vs. Local Metrics**

As we saw in Section 1.4, not all distance functions are created equal. The choice of metric implicitly defines the "energy landscape" the algorithm will explore.

* **Euclidean ($L^2$):** The default metric, $d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum (x_i - y_i)^2}$. It assumes a flat, isotropic space. PCA is a prime example of an algorithm that, by operating on the covariance matrix, implicitly optimizes for this metric.
* **Mahalanobis Distance:** This metric "rescales" the space using the data's covariance, $d_M(\mathbf{x}, \mathbf{y}) = \sqrt{(\mathbf{x}-\mathbf{y})^\top \Sigma^{-1} (\mathbf{x}-\mathbf{y})}$. It is equivalent to computing the Euclidean distance *after* whitening the data (rescaling all principal components to have unit variance). It is a more robust metric that accounts for correlations but is still linear.
* **Cosine Similarity:** This measures the *angle* between two vectors, not their magnitude: $s_C(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|}$. It is ideal for high-dimensional, sparse data (like text or feature histograms) where the *direction* of the vector matters more than its length.

PCA implicitly preserves *global* Euclidean distances. Nonlinear methods like t-SNE and UMAP (Section 3.3) are explicitly designed to preserve *local* distances, building their high-D probability distributions $P_{ij}$ based on near-neighbor relationships.

#### 3.4.2 Curvature Awareness and Distortion

This brings us to a critical point about interpreting nonlinear embeddings: **global distances are not preserved.**

A t-SNE or UMAP plot is *not* a map where the distance between two clusters has a global meaning. These algorithms are designed to preserve local topology at the expense of global geometry. They will "unroll" a manifold, but in doing so, they may "stretch" or "compress" the connections between distant parts.

* **t-SNE:** Famously, t-SNE tends to create well-separated, spherical clusters regardless of the true high-D density. The *size* of a cluster and the *distance between clusters* on a t-SNE plot are often artifacts of the algorithm's t-distribution (Section 3.3) and have no direct physical interpretation (Wattenberg et al., 2016).
* **UMAP:** Tends to preserve more of the global structure than t-SNE, but it still prioritizes local neighborhood connections.

The key takeaway is that these methods are **curvature-aware**. They understand that the true path between points may be curved (the geodesic distance) and attempt to flatten this onto a 2D map. This process *must* introduce distortion, just as a Mercator projection of the Earth distorts the size of Greenland.

#### 3.4.3 Visualization Strategy: Evaluating the Embedding

Since the 2D plot itself can be misleading, how do we evaluate the *quality* of our dimensionality reduction? We must compare the *input* distances to the *output* distances.

A powerful diagnostic is to visualize the **pairwise distance matrix**.
1.  **High-D Matrix ($D_{high}$):** Compute the $N \times N$ matrix of all pairwise distances between our original data points $\{\mathbf{x}_i\}$ using our chosen high-D metric (e.g., Euclidean, RMSD, or correlation distance).
2.  **Low-D Matrix ($D_{low}$):** Compute the $N \times N$ matrix of all pairwise distances between our embedded points $\{\mathbf{z}_i\}$ (e.g., the points on the 2D t-SNE plot).
3.  **Compare:** Visualize $D_{high}$ and $D_{low}$ as heatmaps. If the embedding is good, the two matrices should look similar (or at least share the same block structure). A quantitative comparison can be made by computing the correlation between the elements of $D_{high}$ and $D_{low}$.

This comparison tells us exactly what information was preserved and what was lost in the "projection," allowing us to trust the structures we observe in our low-dimensional map.

---

## **3.5 Clustering — Discovering Groups and Phases**

The dimensionality reduction techniques we've discussed (PCA, UMAP) are designed to *visualize* the data landscape. **Clustering** is the next logical step: it is the formal, algorithmic task of *partitioning* this landscape.

---

### **The Goal: From Point Cloud to Labeled States**

Given a dataset $X = \{\mathbf{x}_i\}$ (either in the full $\mathbb{R}^D$ or, more commonly, in the low-dimensional latent space $\mathbb{R}^d$), clustering aims to assign each point to a specific group, or "cluster." The goal is to find a partition $C = \{C_1, C_2, \dots, C_K\}$ such that:
1.  Points *within* a cluster $C_k$ are highly **similar** to each other (high intra-cluster similarity).
2.  Points in *different* clusters $C_k, C_j$ are highly **dissimilar** (low inter-cluster similarity).

This is a quintessential task of **unsupervised learning**—finding structure in data without any pre-existing labels [5].

---

### **Physical Metaphor: Basins of Attraction**

For the computational physicist, clustering is not just a data-processing step; it is an act of physical discovery. As we established in Chapter 1, the dense regions of our data cloud correspond to the low-energy, high-probability regions of the system's state space.

**Clustering is, therefore, the algorithmic identification of metastable states.**



* **Cluster $C_1$ $\leftrightarrow$ Metastable State 1 (e.g., "Protein Folded")**
```
This is a deep basin in the free-energy landscape. The simulation spends a long time here, so we collect many data points $\{\mathbf{x}_i\}$ in this region, forming a dense cluster.
```
* **Cluster $C_2$ $\leftrightarrow$ Metastable State 2 (e.g., "Protein Unfolded")**
```
This is a separate basin, and thus a separate cluster in the data.
```
* **The "Void" Between Clusters $\leftrightarrow$ The Energy Barrier**
```
The sparse, low-density region between clusters is the high-energy barrier. The clustering algorithm's job is to find this sparse "watershed" and draw a boundary through it.

```
By clustering our simulation data, we are automating the discovery of the system's distinct phases or conformational states.

---

### **A Taxonomy of Clustering Methods**

There is no single "best" clustering algorithm; the best method depends on the *geometry* of the clusters you expect to find. The vast family of algorithms can be grouped by their underlying philosophy [6]:

1.  **Centroid-based (e.g., K-Means):** These algorithms assume a cluster is a "blob" of data whose center, or **centroid**, is a good representative. They work well for discovering convex, spherical-like clusters. We will explore this in Section 3.6.
2.  **Density-based (e.g., DBSCAN):** These algorithms define clusters as continuous regions of high point-density, separated by regions of low density. They are powerful because they can find clusters of *arbitrary shape* (e.g., "S"-shaped or nested rings) and are robust to noise. We will explore this in Section 3.7.
3.  **Connectivity-based (e.g., Hierarchical Clustering):** These methods build a tree (a "dendrogram") of how points are connected. A cluster is defined as a group of points that are "linked" together, either directly or through a chain of neighbors. This approach is explored in Section 3.7.
4.  **Probabilistic (e.g., Gaussian Mixture Models):** These methods assume that the data was generated by a *mixture* of probability distributions (like several distinct Gaussians). Clustering becomes the task of inferring the parameters of these distributions and assigning each point a *probability* of belonging to each cluster. This is the topic of Section 3.8.

??? question "Which Clustering Method Should I Use?"
    The choice depends on your data's geometry and your goals:
    
    * **Spherical, well-separated clusters?** Use K-Means (fast, simple)
    * **Arbitrary shapes or noisy data?** Use DBSCAN (handles complex geometries)
    * **Unknown number of clusters?** Use Hierarchical Clustering (dendrogram reveals structure at all scales)
    * **Need probabilistic assignments?** Use Gaussian Mixture Models (soft clustering with uncertainty)
    * **Very high dimensions?** First reduce dimensionality with UMAP, then cluster in low-D space
    
    Often the best strategy is to try multiple methods and compare results for robustness.
    
---

## **3.6 K-Means — The Simplest Energy Minimizer**

If clustering is the task of finding "basins" in the data landscape (Section 3.5), **K-Means** is the simplest, most direct algorithm for finding them. It is a centroid-based algorithm that models each of the $K$ clusters as a "blob" represented by its center, or **centroid**, $\mathbf{\mu}_k$.

---

### **The Objective Function: A Cluster "Energy"**

K-Means is, at its core, an optimization problem. It defines a simple "energy" or cost function, $J$, and seeks to find the set of $K$ centroids $\{\mathbf{\mu}_k\}$ and the assignment of every point to a cluster that minimizes this cost.

The objective function $J$ is the **intra-cluster variance**, also known as the "sum of squared errors (SSE)":

$$
J = \sum_{i=1}^N \sum_{k=1}^K r_{ik} \|\mathbf{x}_i - \mathbf{\mu}_k\|^2
$$

Here, $N$ is the number of data points, $K$ is the number of clusters, $\mathbf{\mu}_k$ is the centroid of cluster $k$, and $r_{ik}$ is a binary assignment variable:

$$
r_{ik} = \begin{cases} 1 & \text{if data point } i \text{ is assigned to cluster } k \\ 0 & \text{otherwise} \end{cases}
$$

The term $\|\mathbf{x}_i - \mathbf{\mu}_k\|^2$ is the squared Euclidean distance from a point to its assigned centroid. The algorithm's goal is to find the cluster assignments and centroid locations that make the clusters as "tight" and "compact" as possible.

**Physical Analogy:** This objective function is mathematically analogous to finding the minimum energy configuration of $N$ particles (the data points) attracted to $K$ "gravitational centers" (the centroids) via a simple **harmonic potential** (like a spring, $E \propto x^2$).

---

### **The Algorithm: A Two-Step Relaxation**

Minimizing $J$ simultaneously over both the assignments $r_{ik}$ and the centroids $\mathbf{\mu}_k$ is a difficult (NP-hard) problem. K-Means, however, uses a simple and powerful iterative heuristic, known as **Lloyd's Algorithm**, that is guaranteed to find a *local* minimum [5].

This algorithm is a classic example of a **two-step relaxation** or coordinate descent:

1.  **Initialize:** Randomly choose $K$ initial centroids $\mathbf{\mu}_k$ (e.g., by picking $K$ random data points).
2.  **Iterate until convergence:**
    * **(E-Step) Assignment:** Hold the centroids $\{\mathbf{\mu}_k\}$ *fixed*. Assign each data point $\mathbf{x}_i$ to the *nearest* centroid. This step minimizes $J$ with respect to the $r_{ik}$.

$$
r_{ik} \leftarrow 1 \text{ for } k = \arg\min_j \|\mathbf{x}_i - \mathbf{\mu}_j\|^2, \text{ and } r_{ij \neq k} \leftarrow 0
$$

    * **(M-Step) Update:** Hold the assignments $\{r_{ik}\}$ *fixed*. Re-calculate each centroid $\mathbf{\mu}_k$ to be the mean (the center of mass) of all data points currently assigned to it. This step minimizes $J$ with respect to the $\mathbf{\mu}_k$.

$$
\mathbf{\mu}_k \leftarrow \frac{\sum_i r_{ik} \mathbf{x}_i}{\sum_i r_{ik}}
$$

3.  **Converge:** Stop when the assignments no longer change. At this point, the system has "relaxed" into a local minimum of the objective function $J$.

---

### **Limitations**

K-Means is fast and intuitive, but its simplicity leads to two major limitations:
* **Geometric Assumption:** By using the Euclidean distance $\|\mathbf{x}_i - \mathbf{\mu}_k\|^2$, K-Means implicitly assumes that clusters are **convex** and **isotropic** (spherical, non-tilted). It will fail to identify "S"-shaped manifolds, nested rings, or clusters of different densities, as it will try to partition them with simple linear boundaries.
* **Initialization Sensitivity:** The algorithm is only guaranteed to find a *local* minimum of $J$. A poor random initialization can lead to a very poor final clustering. In practice, this is addressed by running the algorithm many times with different random starts (e.g., the "k-means++" initialization) and choosing the run that results in the lowest final $J$ [7].

---

## **3.7 Hierarchical and Density-Based Methods**

---

### **Hierarchical Clustering**

Hierarchical clustering builds a "tree" of cluster relationships, known as a **dendrogram**. The most common approach is **agglomerative (bottom-up) clustering** (Hastie et al., 2009):
1.  **Initialize:** Start with each data point $\mathbf{x}_i$ as its own cluster.
2.  **Iterate:** Find the two "closest" clusters and merge them into a single new cluster.
3.  **Repeat:** Continue this merging process until only one cluster (containing all data points) remains.

The key to this algorithm is the **linkage criterion**, which defines how to measure the "distance" between two *clusters* (not just two points):
* **Single Linkage:** The distance between two clusters is the distance between their *closest* two points. This can connect clusters via a single "bridge" and is good for finding long, "filamentary" structures.
* **Complete Linkage:** The distance is defined by the *farthest* two points. This results in more compact, spherical clusters, as it avoids merging based on a single close pair.
* **Average Linkage:** Uses the average distance between all pairs of points, one from each cluster. This is a common, robust compromise.

**Visualization:** The resulting dendrogram is a powerful visualization tool. We can then "cut" the dendrogram at a chosen height (distance threshold) to produce a flat set of $K$ clusters. The choice of where to cut is a key part of the analysis, allowing the user to explore different scales of structure.

#### 3.7.2 Density-Based Clustering (DBSCAN)

**Density-Based Spatial Clustering of Applications with Noise (DBSCAN)** offers a completely different philosophy. It defines a cluster not by a center, but as a **continuous region of high data-point density**, separated by sparse, low-density gaps (Ester et al., 1996).

DBSCAN is governed by two parameters:
* **$\epsilon$ (epsilon):** A radius defining a "neighborhood" around each point.
* **$minPts$:** The minimum number of points required to be inside a point's $\epsilon$-neighborhood for it to be considered a **core point** (a point in a dense region).

The algorithm proceeds as follows:
1.  A random point $\mathbf{x}_i$ is selected.
2.  If it is a **core point** (has $\ge minPts$ neighbors within $\epsilon$), a new cluster is formed. This cluster "grows" by recursively adding all reachable core points and their neighbors.
3.  If $\mathbf{x}_i$ is a **border point** (fewer than $minPts$ neighbors, but is a neighbor of a core point), it is assigned to that cluster.
4.  If $\mathbf{x}_i$ is neither core nor border, it is labeled as **noise**.

---

### **Advantages and Visualization**

The primary advantage of these methods is their **flexibility**.
* Both hierarchical (with single linkage) and density-based methods can identify clusters of **arbitrary shape**, including the non-convex structures that K-Means fails on.
* DBSCAN is exceptionally robust to **noise**, as it explicitly identifies and isolates outlier points rather than forcing them into a cluster.

In our data-driven physics workflow, the typical use-case is to *first* run a nonlinear embedding like UMAP (Section 3.3) to create a 2D or 3D map, and *then* run an algorithm like DBSCAN (or its more advanced hierarchical version, HDBSCAN) on this low-dimensional map. We can then visualize the result by **overlaying the cluster labels as colors** on the UMAP projection, instantly revealing the distinct phases or metastable states as separate, contiguous regions on our map.

---

## **3.8 Probabilistic and Energy-Based Clustering**

The methods in the previous sections (K-Means, DBSCAN) perform **hard assignments**, where each data point $\mathbf{x}_i$ is definitively assigned to a single cluster $C_k$. This is a rigid, all-or-nothing view. A more flexible and statistically-grounded approach is **probabilistic clustering**, which assumes the data is a *mixture* of underlying probability distributions. The goal is no longer to find hard boundaries, but to infer the parameters of these distributions.

---

### **Gaussian Mixture Models (GMMs)**

The most widely used probabilistic clustering method is the **Gaussian Mixture Model (GMM)**. A GMM assumes that the data is generated from a weighted sum of $K$ different Gaussian distributions (or "blobs"). Each Gaussian $k$ is a cluster, defined by its own mean $\mathbf{\mu}_k$ and covariance matrix $\Sigma_k$.

The probability density of a GMM is [5]:

$$
p(\mathbf{x}) = \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x}|\mathbf{\mu}_k, \Sigma_k)
$$

* $\mathcal{N}(\mathbf{x}|\mathbf{\mu}_k, \Sigma_k)$ is the probability density of a single $D$-dimensional Gaussian cluster.
* $\pi_k$ are the **mixing coefficients**, which are positive and sum to one ($\sum_k \pi_k = 1$). They represent the *prior probability* that a data point was generated from cluster $k$.

Unlike K-Means, which only has centroids $\mathbf{\mu}_k$ and assumes spherical clusters, a GMM's covariance matrices $\Sigma_k$ allow it to model clusters that are **elliptical** and have **different orientations and sizes**.

---

### **The Expectation-Maximization (EM) Algorithm**

The parameters $\mathbf{\theta} = (\{\pi_k, \mathbf{\mu}_k, \Sigma_k\})$ are typically fit using **Expectation-Maximization (EM)**, an iterative algorithm that is a more general version of the K-Means relaxation (Section 3.6). It alternates between two steps [9]:

1.  **E-Step (Expectation):** Hold the current parameters $\mathbf{\theta}$ fixed. For each data point $\mathbf{x}_i$, compute the **posterior probability** (or "responsibility") $p(k|\mathbf{x}_i)$ that it belongs to cluster $k$. This is the "soft assignment."

$$
p(k|\mathbf{x}_i) = \frac{\pi_k \mathcal{N}(\mathbf{x}_i|\mathbf{\mu}_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(\mathbf{x}_i|\mathbf{\mu}_j, \Sigma_j)}
$$

2.  **M-Step (Maximization):** Hold the soft assignments $p(k|\mathbf{x}_i)$ fixed. Update the parameters $\mathbf{\theta}$ to maximize the expected log-likelihood. This re-calculates the mean, covariance, and mixing coefficients for each cluster, weighted by the soft assignments.

These two steps are repeated until the log-likelihood of the data, $\ln \mathcal{L}(\mathbf{\theta}) = \sum_i \ln p(\mathbf{x}_i)$, converges.

---

### **Soft Assignments**

The E-step provides the key advantage of GMMs: **soft assignments**. The posterior probability $p(k|\mathbf{x}_i)$ is a continuous value between 0 and 1 that represents our *uncertainty* about the cluster membership. A point $\mathbf{x}_i$ poised exactly between two clusters might be assigned $p(k=1|\mathbf{x}_i) = 0.5$ and $p(k=2|\mathbf{x}_i) = 0.5$. This is a much more physically realistic model for data near phase boundaries or in high-temperature (noisy) states.

---

### **Energy and Thermodynamic Interpretation**

The GMM/EM framework has a deep connection to statistical physics. The **log-likelihood** of the data, $\ln \mathcal{L}(\mathbf{\theta})$, which EM seeks to maximize, can be directly interpreted as the **negative of the free energy** $\mathcal{F}$ of the system [5].

$$
\mathcal{F} = \langle E \rangle - T S
$$

In this analogy:

* The E-step, which computes the posterior probabilities, is equivalent to finding the optimal distribution of "hidden" variables (the cluster assignments) that *minimizes* the free energy for fixed parameters.
* The M-step, which updates the parameters, is equivalent to updating the "energy function" itself to best match the inferred distribution.

Therefore, the **EM algorithm is a form of free-energy minimization**. This provides a powerful, physics-based justification for the algorithm: it is not just a statistical fitting procedure, but a relaxation process that, like a physical system, settles into an equilibrium state of minimum free energy (maximum likelihood).

---

## **3.9 Worked Example — Discovering Phases in Simulation Data**

This example demonstrates our complete workflow (Chapters 1-3) on a cornerstone problem in statistical physics: identifying the phases of the **2D Ising model**. We will use raw simulation data, apply dimensionality reduction to find the underlying manifold, and use clustering to automatically discover the system's distinct physical phases.

---

### **The System and Data Generation**

* **Physical System:** The 2D Ising model on an $L \times L$ lattice. We run Monte Carlo simulations (as in *Volume II*) at a range of temperatures $T$, from well below the critical temperature $T_c$ (ordered, ferromagnetic) to well above $T_c$ (disordered, paramagnetic).
* **Data (Step 1):** We collect $N$ "snapshots" (spin configurations) from these simulations. Each snapshot is an $L \times L$ matrix of $\pm 1$ spins. We **flatten** each snapshot into a single $D=L^2$ dimensional vector $\mathbf{x}_i \in \mathbb{R}^D$. Our dataset $X$ is a collection of these high-dimensional vectors, each "colored" by the temperature $T$ at which it was generated.

---

### **Dimensionality Reduction (Step 2)**

We cannot use PCA. The relationship between the two ordered phases (all spins up vs. all spins down) and the disordered phase (random spins) is highly **nonlinear**. A spin-up state $\mathbf{x}_{\uparrow}$ and a spin-down state $\mathbf{x}_{\downarrow}$ are "close" in energy but "far" in Euclidean distance.

We must use a manifold learning algorithm. We apply **UMAP** (Section 3.3) to our entire dataset $X$ to project all $N$ configurations from $\mathbb{R}^D$ down to $\mathbb{R}^2$. The resulting 2D coordinates $\{\mathbf{z}_i\}$ represent our learned "map" of the Ising model's state space.

---

### **Clustering (Step 3)**

We now analyze the resulting 2D UMAP projection. The points form distinct "continents" or clusters. We apply a density-based algorithm like **DBSCAN** (Section 3.7) to this 2D space. DBSCAN is ideal here because the clusters are non-spherical and separated by low-density voids.

---

### **Observation and Physical Interpretation**

The visual output—a 2D UMAP scatter plot, with points colored by their DBSCAN cluster label—reveals the system's physics with remarkable clarity:



We typically observe **three distinct, well-separated clusters**:
1.  **Cluster 1 (Ordered, Up):** A dense, tight cluster. Upon inspection, these points correspond to low-temperature ($T < T_c$) snapshots where the average magnetization is $M \approx +1$. This is the **ferromagnetic "spin-up" phase**.
2.  **Cluster 2 (Ordered, Down):** An identical dense, tight cluster. These points correspond to low-temperature snapshots with $M \approx -1$. This is the **ferromagnetic "spin-down" phase**.
3.  **Cluster 3 (Disordered):** A large, more diffuse cluster. These points correspond to high-temperature ($T > T_c$) snapshots where $M \approx 0$. This is the **paramagnetic "disordered" phase**.

The points generated near the critical temperature $T_c$ will often lie in the sparse "voids" or "transition regions" between these three main clusters, representing the critical fluctuations.

**Conclusion:** By treating the simulation output as data, our `UMAP + DBSCAN` pipeline has successfully, and *without any prior knowledge of magnetism*, re-discovered the fundamental structure of the Ising model. It identified the two distinct, low-energy "basins" (the ordered phases) and the high-energy, high-entropy "plateau" (the disordered phase), confirming that clustering is a powerful, data-driven method for identifying and partitioning a system's physical states.

---

## **3.10 Code Demo — PCA + Clustering**

This demonstration provides a practical, end-to-end example of the chapter's workflow. We will:

1.  Generate a synthetic 5-dimensional ($D=5$) dataset that has an inherent, "hidden" cluster structure ($K=3$).
2.  Apply **PCA** (Section 3.2) to reduce this high-dimensional data to a 2-dimensional ($d=2$) space, which we can visualize.
3.  Apply **K-Means clustering** (Section 3.6) to this low-dimensional "map" to algorithmically partition the data.
4.  Visualize the final 2D projection, coloring the points by their discovered cluster labels.

This process simulates the discovery of "phases" or "states" from high-dimensional, unlabeled simulation data.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

## 1. Generate synthetic high-dimensional data

## We create 1000 samples in 5 dimensions (D=5)

## with 3 distinct cluster centers (K=3).

X, y_true = make_blobs(n_samples=1000, WAR: tool-call-rejected
n_features=5,
centers=3,
cluster_std=1.2,
random_state=0)

## X is our (1000, 5) data matrix. y_true holds the "ground truth" labels,

## but we will only use them for validation, not for the clustering.

## 2. Apply PCA for dimensionality reduction

pca = PCA(n_components=2)
## fit_transform finds the 2 principal components and projects

## the 5D data down to a 2D representation.

X_pca = pca.fit_transform(X)

## X_pca is now our (1000, 2) "map" of the data.

## 3. Apply K-Means clustering

## We apply K-Means to the 2D projected data.

kmeans = KMeans(n_clusters=3, random_state=0)
labels = kmeans.fit_predict(X_pca)
## 'labels' is an array of 0s, 1s, and 2s, assigning each point to a cluster.

## 4. Visualization

plt.figure(figsize=(9, 7))
## Create a scatter plot of the PCA-projected data (PC1 vs PC2)

## Color each point (c=labels) according to its discovered cluster.

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)

plt.title('PCA Projection + K-Means Clustering')
plt.xlabel('Principal Component 1 (PC1)')
plt.ylabel('Principal Component 2 (PC2)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
```

**Interpretation:**
The resulting plot shows the 2D principal component "map" of our original 5D data. The K-Means algorithm, operating *only* on this 2D data, has successfully identified the three distinct groups. The `cmap='viridis'` coloring clearly visualizes these discovered partitions.

This two-step process—`PCA -> K-Means`—is a powerful and common workflow. PCA first "unveils" the dominant structure by projecting away the noise and redundancy of high-dimensional space, and K-Means then automatically partitions this simplified map into its most prominent, high-density regions.

---

## **3.11 Takeaways & Bridge to Chapter 4**

---

### **What We Accomplished in Chapter 3**

This chapter provided the complete, data-driven workflow for moving from a high-dimensional data cloud (Chapter 1) to an interpretable, low-dimensional map of physical states.

* **Dimensionality Reduction Finds the Manifold:** We saw that high-dimensional data is a "curse" (Chapter 2), but that physical data is constrained to a low-dimensional manifold. We used **PCA** to find the optimal *linear* projection of this manifold (the collective, physical modes) and **nonlinear methods (UMAP/t-SNE)** to "unroll" the manifold's *curved* geometry, preserving its local topology.

* **Clustering Finds the Phases:** We then treated this low-dimensional map as a data-driven "free energy landscape." We applied **clustering algorithms**—from the energy-minimizing **K-Means** to the density-based **DBSCAN** and the probabilistic **GMM**—to algorithmically partition this landscape.

* **Structure from Data:** The final result is a powerful synthesis. By combining `UMAP -> DBSCAN` (as in our Ising example), we turned an incomprehensible $D$-dimensional trajectory file into a simple, visual map of the system's core physics: the distinct clusters *are* the system's metastable states or phases. We have successfully turned unstructured data into interpretable structure: **coordinates (from DR) + states (from clustering)**.

---

### **Bridge to Part II: Optimization as Physics**

In Part I (Chapters 1-3), our perspective was that of an **observer**. We analyzed a *static* dataset that was the *result* of a simulation. We were, in essence, data-driven cartographers mapping a fixed, unknown landscape.

Now, in **Part II**, we change our perspective entirely. We become the **agent**. We are no longer *given* a data cloud that has already sampled the basins; instead, we are placed *on* an energy/loss landscape and must *find* the basin ourselves.

The landscapes we just learned to *identify* are the same landscapes we must now learn to *navigate*. The task shifts from **analysis** to **search**.

---

## **References**

[1] Amadei, A., Linssen, A. B., & Berendsen, H. J. (1993). Essential Dynamics of Proteins. *Proteins: Structure, Function, and Bioinformatics*, 17(4), 412-425.

[2] Maaten, L. van der, & Hinton, G. (2008). Visualizing Data using t-SNE. *Journal of Machine Learning Research*, 9, 2579-2605.

[3] McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv:1802.03426*.

[4] Wattenberg, M., Viégas, F., & Johnson, I. (2016). How to Use t-SNE Effectively. *Distill*, 1(10), e2.

[5] Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.

[6] Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.

[7] Arthur, D., & Vassilvitskii, S. (2007). k-means++: The Advantages of Careful Seeding. *Proceedings of the 18th Annual ACM-SIAM Symposium on Discrete Algorithms (SODA)*.

[8] Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. *Proceedings of the 2nd International Conference on Knowledge Discovery and Data Mining (KDD-96)*.

[9] Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum Likelihood from Incomplete Data via the EM Algorithm. *Journal of the Royal Statistical Society, Series B*, 39(1), 1-38.

* In Chapter 3, a "cluster" was a low-energy basin we *found*.
* In Chapter 4, "The Optimization Landscape," a "minimum" is a low-energy basin we must *seek*.

We will take the physical analogies we've built—energy landscapes, basins of attraction, and barrier-crossing—and make them the central object of study. We will see that **every machine learning algorithm is, at its core, a physical system** (an optimizer) following a set of dynamical laws (e.g., gradient descent) as it "rolls" downhill on a high-dimensional loss surface, seeking equilibrium at the bottom.
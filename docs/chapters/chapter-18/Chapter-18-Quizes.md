# **Chapter 18 : Quizes**

---

!!! note "Quiz"
    **1. What is the primary function of the normalized adjacency matrix, $\tilde{\mathbf{A}}$, in a Graph Convolutional Network (GCN)?**

    - A. To store the raw connectivity of the graph.
    - B. To perform a normalized averaging of features from neighboring nodes, including self-features.
    - C. To learn the weights for feature transformations.
    - D. To apply a non-linear activation function to node features.

    ??? info "See Answer"
        **Correct: B**

        *(The matrix $\tilde{\mathbf{A}} = \hat{\mathbf{D}}^{-1/2} \hat{\mathbf{A}} \hat{\mathbf{D}}^{-1/2}$ is specifically designed to prevent the scale of feature vectors from exploding by averaging, not just summing, the features from a node's local neighborhood.)*

---

!!! note "Quiz"
    **2. In the GCN layer equation $\mathbf{H}^{(l+1)} = \sigma(\tilde{\mathbf{A}} \mathbf{H}^{(l)} \mathbf{W}^{(l)})$, what does the operation $\tilde{\mathbf{A}} \mathbf{H}^{(l)}$ represent?**

    - A. Feature transformation and dimensionality reduction.
    - B. The message passing or local aggregation step.
    - C. The application of a non-linear activation.
    - D. The calculation of the graph's Laplacian.

    ??? info "See Answer"
        **Correct: B**

        *(This matrix multiplication is the core of the "convolution" where each node's feature vector becomes a blend of its own and its neighbors' features from the previous layer.)*

---

!!! note "Quiz"
    **3. Why are self-loops typically added to the adjacency matrix ($\mathbf{A} + \mathbf{I}$) before normalization in a GCN?**

    - A. To ensure the graph is undirected.
    - B. To increase the sparsity of the graph.
    - C. To ensure that a node's own features are included in the aggregation for the next layer.
    - D. To make the degree matrix invertible.

    ??? info "See Answer"
        **Correct: C**

        *(Without a self-loop, a node's new representation would be based solely on its neighbors, completely ignoring its own prior state.)*

---

!!! note "Quiz"
    **4. The iterative application of the GCN propagation rule, $\mathbf{H}^{(l+1)} = \tilde{\mathbf{A}} \mathbf{H}^{(l)}$, without non-linearities or weights, leads to what phenomenon?**

    - A. Overfitting to the training data.
    - B. Feature amplification and divergence.
    - C. Over-smoothing, where all node features converge to the same value.
    - D. The creation of a deep, hierarchical feature representation.

    ??? info "See Answer"
        **Correct: C**

        *(Repeatedly averaging with neighbors causes initial feature differences to "diffuse" across the graph, eventually leading to a uniform state where nodes are indistinguishable.)*

---

!!! note "Quiz"
    **5. How is the GCN's message passing mechanism analogous to discrete heat diffusion?**

    - A. Both processes are driven by learnable weight matrices.
    - B. The normalized adjacency matrix acts as a smoothing filter, spreading a high feature value (like "heat") to its neighbors over successive layers.
    - C. Both require a non-linear activation function to propagate information.
    - D. The process only works on grid-like graph structures.

    ??? info "See Answer"
        **Correct: B**

        *(Just as heat flows from a hot region to cooler ones until equilibrium is reached, the GCN's aggregation step averages out localized high feature values across the graph.)*

---

!!! note "Quiz"
    **6. What fundamental limitation of GCNs does the Graph Attention Network (GAT) address?**

    - A. GCNs cannot process graphs with weighted edges.
    - B. GCNs assign a fixed, structurally-determined importance to all neighbors during aggregation.
    - C. GCNs are computationally too expensive for large graphs.
    - D. GCNs can only be applied to undirected graphs.

    ??? info "See Answer"
        **Correct: B**

        *(GATs overcome this by introducing an attention mechanism that allows the model to learn to assign different weights (importance) to different neighbors.)*

---

!!! note "Quiz"
    **7. In a GAT, how are the attention coefficients ($\alpha_{ij}$) calculated?**

    - A. They are a fixed property of the graph structure, derived from the degree matrix.
    - B. They are calculated by normalizing compatibility scores between node pairs using a softmax function.
    - C. They are directly optimized as standalone parameters for each pair of nodes.
    - D. They are determined by the shortest path distance between nodes.

    ??? info "See Answer"
        **Correct: B**

        *(A scoring function first computes a value $e_{ij}$ based on the features of nodes $i$ and $j$, and these scores are then passed through a softmax across all of $i$'s neighbors to produce the final attention weights $\alpha_{ij}$.)*

---

!!! note "Quiz"
    **8. What is the role of the trainable attention vector $\mathbf{a}$ in a GAT?**

    - A. It is the final feature representation of a node.
    - B. It is a non-linear activation function.
    - C. It is used to compute the compatibility score $e_{ij}$ between two nodes' features.
    - D. It normalizes the adjacency matrix.

    ??? info "See Answer"
        **Correct: C**

        *(The score is typically calculated as $e_{ij} = \text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}\mathbf{h}_i || \mathbf{W}\mathbf{h}_j])$, meaning $\mathbf{a}$ is a learned parameter that helps determine how compatible or important two connected nodes are to each other.)*

---

!!! note "Quiz"
    **9. The concept of "message passing" in GNNs is a generalization of what physical idea?**

    - A. Quantum entanglement.
    - B. Local interaction and influence, as seen in systems like the Ising model or cellular automata.
    - C. Gravitational attraction between distant bodies.
    - D. The Doppler effect in wave propagation.

    ??? info "See Answer"
        **Correct: B**

        *(In these systems, the state of an entity (like a spin) is determined by the states of its immediate neighbors, and this local rule gives rise to global, emergent behavior, just as in GNNs.)*

---

!!! note "Quiz"
    **10. In the Ising model analogy for GNNs, what does the "local field" ($h_i = J \sum_{j \in \mathcal{N}(i)} s_j$) represent?**

    - A. The final, global state of the system.
    - B. The aggregated "message" or influence that a node's neighbors exert on it.
    - C. A random noise term.
    - D. The learnable weight matrix of a GNN layer.

    ??? info "See Answer"
        **Correct: B**

        *(The local field sums the states of the neighboring spins, providing a net influence that guides the central spin $s_i$ to flip or stay, analogous to how a GNN node updates its state based on aggregated neighbor features.)*

---

!!! note "Quiz"
    **11. What does it mean for a GNN to be "equivariant"?**

    - A. The GNN's output is unchanged if the input graph is rotated or translated.
    - B. If the input node features are permuted, the output node features are permuted in the same way.
    - C. The GNN can process graphs of any size.
    - D. The GNN assigns equal importance to all nodes.

    ??? info "See Answer"
        **Correct: B**

        *(This means the network's operation respects the graph's symmetries; re-labeling the nodes leads to a correspondingly re-labeled output.)*

---

!!! note "Quiz"
    **12. What is the primary difference between a Graph Convolutional Network (GCN) and a Graph Hamiltonian Network?**

    - A. GCNs use message passing, while Hamiltonian Networks do not.
    - B. Hamiltonian Networks are based on continuous-time dynamics defined by a graph Hamiltonian, while GCNs use discrete layer-wise updates.
    - C. GCNs can only be used for node classification, while Hamiltonian Networks are for link prediction.
    - D. Hamiltonian Networks are a type of Graph Attention Network.

    ??? info "See Answer"
        **Correct: B**

        *(They model feature evolution as a physical process ($\frac{d\mathbf{H}}{dt} = -i[\mathcal{H}, \mathbf{H}]$), connecting GNNs to concepts from quantum mechanics and dynamical systems.)*

---

!!! note "Quiz"
    **13. What is the "receptive field" of a node in a GNN after $k$ layers of message passing?**

    - A. The node's own initial features.
    - B. The set of all nodes in the graph.
    - C. The set of nodes within a $k$-hop neighborhood of the central node.
    - D. The set of nodes directly connected to the central node (1-hop neighbors).

    ??? info "See Answer"
        **Correct: C**

        *(Each layer of message passing expands the receptive field by one hop, meaning a node's representation after $k$ layers is influenced by information from nodes up to $k$ steps away.)*

---

!!! note "Quiz"
    **14. Which of the following is a common application of Graph Neural Networks?**

    - A. Image classification for non-structured images.
    - B. Natural Language Processing for sequential text.
    - C. Recommender systems, where users and items form a bipartite graph.
    - D. Time-series forecasting for a single data stream.

    ??? info "See Answer"
        **Correct: C**

        *(GNNs are excellent for modeling relationships, making them suitable for tasks like predicting user-item interactions, as well as social network analysis, molecular property prediction, and traffic forecasting.)*

---

!!! note "Quiz"
    **15. In the context of GNNs, what is an "inductive" learning task?**

    - A. A task where the model is trained and tested on the exact same set of nodes.
    - B. A task where the model must generalize to entirely new, unseen graphs or nodes after training.
    - C. A task that involves proving mathematical theorems.
    - D. A task limited to predicting properties of the entire graph (graph-level prediction).

    ??? info "See Answer"
        **Correct: B**

        *(Inductive models like GraphSAGE and GAT learn functions that can be applied to nodes regardless of whether they were seen during training.)*

---

!!! note "Quiz"
    **16. How does the GraphSAGE algorithm differ from a standard GCN?**

    - A. It uses attention to weigh neighbors.
    - B. It uses a fixed, non-learnable aggregator.
    - C. It defines a general, trainable aggregation function (e.g., mean, LSTM, or max-pooling) to sample and aggregate features from a node's local neighborhood.
    - D. It can only be used on directed graphs.

    ??? info "See Answer"
        **Correct: C**

        *(This makes it powerful for inductive learning on large graphs where using the full neighborhood is infeasible.)*

---

!!! note "Quiz"
    **17. The normalization term $\hat{\mathbf{D}}^{-1/2}$ in the GCN formula is used to:**

    - A. Add non-linearity to the model.
    - B. Control for nodes with large degrees that would otherwise dominate the feature aggregation process.
    - C. Ensure the weight matrix $\mathbf{W}$ remains orthogonal.
    - D. Convert the graph to a directed graph.

    ??? info "See Answer"
        **Correct: B**

        *(By dividing by the (square root of the) degree, it ensures that the influence of neighbors is averaged, preventing the features of high-degree nodes from disproportionately affecting their neighbors.)*

---

!!! note "Quiz"
    **18. What is a primary motivation for using GNNs to model physical systems?**

    - A. Physical laws often depend on local interactions between entities, which maps directly to the message-passing paradigm.
    - B. GNNs are the only models capable of handling large datasets.
    - C. All physical systems can be represented as grid-like structures.
    - D. GNNs inherently conserve energy and momentum.

    ??? info "See Answer"
        **Correct: A**

        *(The structure of a GNN, where a node's state is updated based on its neighbors, is a natural fit for modeling systems governed by local rules, like particle interactions or spin systems.)*

---

!!! note "Quiz"
    **19. In the GAT attention calculation, what is the purpose of the LeakyReLU activation function often applied to the compatibility score $e_{ij}$?**

    - A. To ensure the final attention coefficients sum to one.
    - B. To introduce non-linearity into the attention mechanism itself.
    - C. To make the scores strictly positive.
    - D. To sparsify the attention weights, setting some to zero.

    ??? info "See Answer"
        **Correct: B**

        *(This allows the model to learn more complex relationships when determining the importance of one node to another, beyond a simple linear combination.)*

---

!!! note "Quiz"
    **20. A GNN designed for a graph-level prediction task (e.g., predicting a molecule's toxicity) must include what additional component?**

    - A. More GNN layers.
    - B. An attention mechanism.
    - C. A "readout" or global pooling layer to aggregate all node features into a single graph representation.
    - D. Edge features in addition to node features.

    ??? info "See Answer"
        **Correct: C**

        *(After the GNN layers produce final node embeddings, a readout function (like global mean/sum pooling or a more sophisticated method) is needed to combine them into a fixed-size vector for the entire graph.)*

---

!!! note "Quiz"
    **21. The "message" a node $j$ sends to a node $i$ in a simple GCN is typically:**

    - A. A complex function of the entire graph structure.
    - B. The feature vector of node $j$ from the previous layer, $\mathbf{h}_j^{(l)}$, scaled by a normalization constant.
    - C. A learnable parameter specific to the edge $(i, j)$.
    - D. The shortest path distance between $i$ and $j$.

    ??? info "See Answer"
        **Correct: B**

        *(The core idea is that neighbors "pass" their current state (feature vector) to the central node for aggregation.)*

---

!!! note "Quiz"
    **22. What is a potential downside of making a GNN very deep (i.e., stacking many layers)?**

    - A. It becomes unable to learn from graph structures.
    - B. It leads to the problem of over-smoothing, making node representations indistinguishable.
    - C. The computational cost decreases with each added layer.
    - D. It can only be applied to smaller graphs.

    ??? info "See Answer"
        **Correct: B**

        *(As the receptive field of each node expands to include almost the entire graph, the local structural information is lost, and all nodes converge to a similar feature vector.)*

---

!!! note "Quiz"
    **23. The adjacency matrix $\mathbf{A}$ of a graph is a representation of its:**

    - A. Node features.
    - B. Topology or connectivity.
    - C. Degree distribution.
    - D. Laplacian spectrum.

    ??? info "See Answer"
        **Correct: B**

        *(The matrix element $A_{ij}$ is 1 if there is an edge connecting node $i$ and node $j$, and 0 otherwise, thus encoding the direct relationships between nodes.)*

---

!!! note "Quiz"
    **24. In the GAT conceptual code, why are the compatibility scores ($e_{ij}$) passed through a softmax function?**

    - A. To make the scores negative.
    - B. To convert the arbitrary scores into a normalized probability distribution (the attention weights $\alpha_{ij}$) that sums to 1.
    - C. To increase the magnitude of the scores.
    - D. To select only the neighbor with the highest score and ignore all others.

    ??? info "See Answer"
        **Correct: B**

        *(This ensures that the aggregation is a weighted average, where the weights reflect the relative importance of each neighbor.)*

---

!!! note "Quiz"
    **25. The success of GNNs is largely due to their ability to combine:**

    - A. Unsupervised and supervised learning.
    - B. Information from the graph's structure (topology) with node/edge features (attributes).
    - C. Convolutional and recurrent neural network architectures.
    - D. Continuous and discrete optimization methods.

    ??? info "See Answer"
        **Correct: B**

        *(GNNs learn representations by propagating and transforming feature information according to the underlying graph connectivity, effectively merging these two sources of information.)*

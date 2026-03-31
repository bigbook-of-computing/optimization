# **Chapter 11: Graphical Models & Probabilistic Graphs**

---

# **Introduction**

The inference frameworks of Chapters 9 and 10 treated variables largely in isolation—posterior distributions over individual parameter vectors, linear models mapping independent features to outputs. But real-world systems, from molecular interactions and neural circuits to social networks and image pixels, exhibit **structured interdependencies**: variables are not independent but coupled through networks of conditional relationships. A protein's expression depends on regulatory genes, a neuron's firing depends on its synaptic neighbors, a pixel's label correlates with surrounding pixels. This chapter extends Bayesian inference from single models to **Probabilistic Graphical Models (PGMs)**—mathematical frameworks that use graph topology to encode, visualize, and exploit conditional independence structures, transforming high-dimensional joint distributions into tractable products of local conditional probabilities.

We begin by motivating the need for structured representations: modeling $N$ interdependent variables naively requires specifying an exponentially large joint probability table, but graphical factorizations reduce this to manageable local terms. **Bayesian Networks (BNs)** use directed acyclic graphs (DAGs) to encode causal hierarchies, factorizing joint distributions via the chain rule $p(\mathbf{x}) = \prod_i p(x_i|\text{parents}(x_i))$—each node depends only on its direct ancestors, not the entire history. **Markov Random Fields (MRFs)** use undirected graphs to model symmetric interactions (mutual dependencies without causality), factorizing distributions through **potential functions over cliques** and yielding the Gibbs distribution $p(\mathbf{x}) = \frac{1}{Z} e^{-E(\mathbf{x})}$—the direct generalization of the Ising model to arbitrary graph topologies. The concept of **conditional independence** and **Markov Blankets** reveals that each variable is statistically decoupled from the rest of the network given its immediate neighbors, enabling local computation to achieve global inference. **Factor Graphs** provide a unified bipartite representation separating variables from coupling functions, simplifying the implementation of inference algorithms. The centerpiece is **Belief Propagation (BP)**—an iterative message-passing algorithm where nodes exchange local probability summaries until the system converges to statistical equilibrium (exact for trees, approximate for loopy graphs). We connect inference to optimization through **Variational Inference**, showing that approximating intractable posteriors via simpler distributions $q$ is equivalent to minimizing the KL divergence or, equivalently, the **Variational Free Energy** $\mathcal{F}(q)$—the mean-field approximation of statistical physics. Extensions to **Dynamic Bayesian Networks** (DBNs) handle temporal sequences, with Hidden Markov Models (HMMs) and Kalman Filters modeling state evolution and observation noise over time.

By the end of this chapter, you will understand graphical models as the structural language of inference: DAGs encode causal flow (information propagates along directed edges), MRFs encode energy landscapes (equilibrium distributions over symmetric couplings), and Factor Graphs make factorizations explicit for algorithmic implementation. You will see that Belief Propagation is distributed relaxation to statistical equilibrium (messages as local forces, convergence as collective consistency), Variational Inference is free energy minimization (approximate posteriors as mean-field theories), and learning graphical models is structural discovery (inferring both coupling strengths and network topology from data). Worked examples demonstrate BP on simple binary chains (iterative message updates converging to exact marginals), and code implementations visualize the relaxation dynamics. These foundations unify statistics, physics, and computation: MRFs generalize Ising models to arbitrary graphs, BP mirrors spin relaxation, and variational methods connect inference to thermodynamic optimization. This completes Part III's exploration of learning as inference, preparing us for Part IV (Deep Learning as Representation), where massively parameterized neural networks learn hierarchical feature transformations autonomously, replacing handcrafted graphical structures with learned internal representations that simplify complex probability landscapes.

---

# **Chapter 11: Outline**

| **Sec.** | **Title** | **Core Ideas & Examples** |
|:---|:---|:---|
| **11.1** | From Single Models to Networks of Belief | Structured dependencies: variables coupled through networks (molecules, neurons, pixels); naive joint $p(x_1, \dots, x_N)$ exponentially large; graphical factorization: $p(\mathbf{x}) = \prod_i p(x_i|\text{parents}(x_i))$ (directed) or $\prod_C \psi_C(\mathbf{x}_C)$ (undirected); directed (Bayesian Networks, causal flow) vs undirected (Markov Random Fields, symmetric interactions); local relationships $\to$ global inference; Example: graph as blueprint of interaction |
| **11.2** | Bayesian Networks — Directed Acyclic Graphs (DAGs) | DAG structure: nodes (variables), directed edges (conditional dependencies), acyclicity (causal hierarchy); joint factorization: $p(\mathbf{x}) = \prod_i p(x_i|\text{parents}(x_i))$ (chain rule); inference: diagnostic (backward) and causal (forward) message passing; conditional probability distributions (CPDs/CPTs); physical analogy: directed energy flow, unidirectional causality; Example: chain $A \to B \to C$, $p(A,B,C) = p(A)p(B|A)p(C|B)$ |
| **11.3** | Markov Random Fields (MRFs) | Undirected graph: symmetric edges (mutual dependencies, no causality); factorization over cliques: $p(\mathbf{x}) = \frac{1}{Z} \prod_C \psi_C(\mathbf{x}_C)$ (potential functions); Gibbs distribution: $p(\mathbf{x}) = \frac{1}{Z} e^{-E(\mathbf{x})}$ (energy form $\psi_C = e^{-E_C}$); partition function $Z$ (normalization); physical analogy: spin networks (Ising/Potts models), coupling energy $J_{ij}$; Example: 2D lattice Ising as MRF with pairwise cliques |
| **11.4** | Conditional Independence and Markov Blankets | Conditional independence: $p(A,B|C) = p(A|C)p(B|C)$ (decoupling given $C$); Markov Blanket: minimal set rendering $x_i$ independent of rest; directed (parents + children + co-parents) vs undirected (direct neighbors); physical analogy: local equilibrium, screening of distant forces (nearest neighbors in Ising); computational implication: local computation suffices for global inference; Example: central spin influenced only by nearest neighbors |
| **11.5** | Factor Graphs — Unified Representation | Bipartite graph: variable nodes (random variables) + factor nodes (local functions $f_a$); factorization: $p(\mathbf{x}) = \frac{1}{Z} \prod_a f_a(\mathbf{x}_a)$ (explicit product); benefit: separates variables from coupling functions, simplifies message passing; physical analogy: variables as particles, factors as interaction sites/energy functions; Example: unified framework for BNs and MRFs |
| **11.6** | Belief Propagation (BP) | Iterative message passing: variable-to-factor $m_{i \to a}$ and factor-to-variable $m_{a \to i}$ messages; goal: compute marginal probabilities $p(x_i)$ (updated beliefs); convergence: exact for trees (no cycles), approximate for loopy graphs; message update: $m_{i \to a}(x_i) = \prod_{k \in N(i) \setminus a} m_{k \to i}(x_i)$ (local accumulation); physical analogy: distributed relaxation to statistical equilibrium, collective consistency; Example: BP as miniature Ising relaxation |
| **11.7** | Variational Inference and Free Energy Minimization | Intractable posterior $p(\mathbf{\theta}|\mathcal{D})$ approximated by tractable $q(\mathbf{\theta})$; minimize KL divergence: $\min_q D_{\mathrm{KL}}(q\|p)$; variational free energy (ELBO): $\mathcal{F}(q) = \mathbb{E}_q[\ln q - \ln p(\mathcal{D}, \mathbf{\theta})]$; mean-field approximation: decouple interactions (independent factors in $q$); physical analogy: VI as mean-field theory (simplified spin-glass approximation); Example: EM algorithm, inference as energy minimization |
| **11.8** | Loopy and Approximate Belief Propagation | BP exact only for trees; loopy graphs (cycles): messages recirculate, statistical correlations violate independence assumptions; Loopy BP (LBP): apply BP to cyclic graphs (approximate algorithm); convergence: not guaranteed, but often effective (Bethe approximation to free energy); physical interpretation: loops as frustrated spins (conflicting local energies), long-range correlations; Example: image denoising, error-correcting codes |
| **11.9** | Dynamic and Temporal Graphical Models | Dynamic Bayesian Networks (DBNs): sequences of graphs linked across time; Markov property: $p(\mathbf{x}_t|\mathbf{x}_{1:t-1}) = p(\mathbf{x}_t|\mathbf{x}_{t-1})$ (future depends only on present); factorization: $p(\mathbf{x}_1)$ (initial state), $p(\mathbf{x}_t|\mathbf{x}_{t-1})$ (transitions), $p(\mathbf{y}_t|\mathbf{x}_t)$ (observations); Hidden Markov Models (HMMs): hidden states $\mathbf{z}_t$, observations $\mathbf{x}_t$; Kalman Filter: continuous linear-Gaussian case; physical analogy: system evolution (equations of motion), state estimation (filtering noise); Example: sequence tagging, robotics localization |
| **11.10** | Worked Example — Belief Propagation on a Simple Graph | Binary chain $A - B - C$; goal: compute marginal $p(B)$; potentials: $\psi_{AB}(A,B)$, $\psi_{BC}(B,C)$; message passing: $A \to B$ and $C \to B$ send influence summaries; final belief: $\text{Belief}(B) \propto m_{A \to B} \cdot m_{C \to B}$; exact solution (tree): single pass convergence; interpretation: local updates $\to$ global inference, Ising relaxation analogy; Example: miniature spin alignment to equilibrium |
| **11.11** | Code Demo — Belief Propagation on a Binary Chain | Python implementation: pairwise potentials $\psi_{AB}$, $\psi_{BC}$ (coupling matrices); initialize uniform messages; iterative updates: factor-to-variable and variable-to-factor message passing; normalization to prevent overflow; final belief computation: $\text{belief}_B = m_{A \to B} \cdot m_{C \to B}$; interpretation: relaxation dynamics (messages stabilize), local forces achieve global consistency; Example: visualizing BP convergence |
| **11.12** | Graphical Models and Learning | Parameter learning: infer CPD values or potential function weights given fixed structure; maximize log-likelihood $\ln p(\mathcal{D}|\mathcal{G}, \mathbf{\theta})$; EM algorithm for latent variables (iterative free energy minimization); structure learning: infer graph topology (edges/connectivity); constraint-based (conditional independence tests) vs score-based (maximize model evidence); physical analogy: inferring force strengths (parameter) vs connectivity (structure); Example: discovering regulatory networks, physical interaction graphs |
| **11.13** | Applications Across Domains | Physics: MRFs/Ising models (spin correlations, ground state search); Computer Vision: CRFs (image segmentation/denoising, smoothness constraints); Genomics: BNs (gene regulation pathways, causal inference $A \to B \to C$); NLP: HMMs/CRFs (sequence tagging, part-of-speech); Robotics: DBNs/Kalman Filters (sensor fusion, localization); unifying concept: graph as interaction blueprint, local rules $\to$ global behavior; Example: universal framework across physics, engineering, data science |
| **11.14** | Takeaways & Bridge to Part IV | Structure unifies science: graphs encode dependencies (statistics + physics + computation); inference as dynamics: BP is distributed relaxation to statistical equilibrium (Markov Blanket consistency $\to$ global truth); variational inference as free energy minimization (mean-field approximation); learning is structural (infer forces and topology from data); bridge to Part IV: from handcrafted graphical models to learned deep hierarchies (neural networks learn internal representations autonomously); Example: fixed probabilistic structures $\to$ massively parameterized feature transformations |

---

## **11.1 From Single Models to Networks of Belief**

---

### **Motivation: Structured Dependencies**

In the real world—from the interactions between atoms in a molecule to the firing patterns of neurons in a brain—variables are rarely independent. Modeling such a system by treating each variable separately would be naive and highly inefficient.

* **Real-world Systems:** Molecules, neurons, and words all form **structured dependencies**.
* **Need for Structure:** We need a mathematical language that explicitly describes which variables are coupled and how that coupling determines the overall system state.

---

### **Core Idea: Representing Joint Probability via Graph**

Graphical models solve this by leveraging the insights of graph theory (nodes and edges) to represent complex joint probability distributions.

The graph structure acts as a "blueprint" for factorizing the joint probability distribution $p(\mathbf{x}) = p(x_1, \dots, x_N)$ into a product of simpler, local probability terms. For **directed graphs** (Bayesian Networks), this factorization is given by the chain rule applied to the graph's structure:

$$
p(\mathbf{x}) = \prod_i p(x_i|\text{parents}(x_i))
$$

### Two Main Types: Directed vs. Undirected

Graphical models are broadly categorized based on the nature of the dependency encoded by the edges:

* **Directed Graphs $\rightarrow$ Bayesian Networks (BNs):** Edges have a direction ($\to$), encoding a clear **causal hierarchy** or flow of influence.
* **Undirected Graphs $\rightarrow$ Markov Random Fields (MRFs):** Edges are symmetric, encoding **mutual dependencies** or symmetric interactions without implying direction or causality.

| Type | Analogy | Primary Use |
| :--- | :--- | :--- |
| **Bayesian Nets (Directed)** | **Flow of information**. | Modeling causal reasoning and sequence prediction. |
| **Markov Fields (Undirected)** | **Exchange of forces**. | Modeling physical interactions and image segmentation. |

---

### **The Goal: Local Relationships to Global Behavior**

The central goal of graphical modeling is to use the encoded **local relationships** (the interactions between neighbors) to successfully perform **global inference**. By defining simple, local rules, we can infer the marginal probabilities and dependencies of a macroscopic system, a principle directly analogous to how local coupling energy defines the global behavior of a physical system.

## **11.2 Bayesian Networks — Directed Acyclic Graphs (DAGs)**

**Bayesian Networks (BNs)**, or Directed Graphical Models, provide a formal way to represent the **causal and informational structure** of a system using a directed graph. They are the foundation for modeling sequential processes and explicit dependency hierarchies.

---

### **Structure: The Directed Acyclic Graph (DAG)**

A Bayesian Network is defined by a **Directed Acyclic Graph (DAG)**, $\mathcal{G} = (\mathcal{V}, \mathcal{E})$:
* **Nodes ($\mathcal{V}$):** Represent the random variables in the system.
* **Directed Edges ($\mathcal{E}$):** Define the conditional dependencies between the variables. An edge from node $A$ to node $B$ ($A \to B$) means that $A$ is a **parent** of $B$, and $B$ is conditionally dependent on $A$.
* **Acyclicity:** The graph must contain **no directed cycles**. This is crucial as it enforces a clear, unambiguous **causal order** or hierarchy within the system.

---

### **Joint Factorization: The Chain Rule**

The power of the BN lies in how its structure simplifies the complex, high-dimensional **joint probability distribution** $p(\mathbf{x}) = p(x_1, \dots, x_N)$. The entire joint distribution can be factored into a product of simpler, local conditional probability distributions (CPDs):

$$
p(\mathbf{x}) = \prod_i p(x_i | \text{parents}(x_i))
$$

* **Example:** For a simple chain graph $A \to B \to C$:

$$
p(A, B, C) = p(A) p(B|A) p(C|B)
$$
```
This factorized form is exponentially simpler to specify and compute than a massive joint probability table.

```
### Inference and Causal Reasoning

The primary task in a BN is **inference**: calculating the probability of unobserved variables given observed evidence.

* Given an observation (evidence) at a node, that information propagates backward (diagnostic inference) and forward (causal inference) throughout the network, updating the posterior probabilities of all other nodes.
* The entire process is viewed as **message passing** or **belief propagation** (Chapter 11.6) along the directed edges of the graph.

---

### **Physical Analogy: Directed Flow of Causality**

In physics, the flow of energy or information often exhibits directionality.

* **Directed Flow:** A Bayesian Network models this as the **directed flow of causality**. For instance, in thermodynamics, heat flows from hot to cold.
* **Unidirectional Energy Transfer:** The graph structure is analogous to a system where influence or "energy" transfer is explicitly unidirectional, defining a clear hierarchical relationship that guides the inference dynamics.

The BN provides the architecture for reasoning under uncertainty where **cause and effect** are integral to the system structure.

## **11.3 Markov Random Fields (MRFs)**

While Bayesian Networks (Section 11.2) model **directed dependencies** with a causal hierarchy, **Markov Random Fields (MRFs)**, or Undirected Graphical Models, represent **symmetric interactions** where the variables are coupled like a network of interacting physical entities. MRFs are the natural statistical language for systems in equilibrium, generalizing the Ising and Potts models from physics.

---

### **Structure: Undirected Graph and Symmetric Dependencies**

An MRF is defined by an **undirected graph** $\mathcal{G} = (\mathcal{V}, \mathcal{E})$.
* **Edges:** An undirected edge between nodes $i$ and $j$ simply indicates that the variables $x_i$ and $x_j$ are directly related or **mutually dependent**.
* **No Causality:** Because the edges have no direction, MRFs explicitly contain no notion of "cause" or flow; they only encode **correlation**.

---

### **Factorization: Potentials over Cliques**

The joint probability distribution $p(\mathbf{x})$ in an MRF is defined not by simple conditional probabilities, but by **potential functions ($\psi_C$)** defined over the graph's **cliques** (fully connected subgraphs).

The joint distribution is factored as a product of these potential functions:

$$
p(\mathbf{x}) = \frac{1}{Z} \prod_{C\in \mathcal{C}} \psi_C(\mathbf{x}_C)
$$
* $\mathcal{C}$ is the set of maximal cliques in the graph.
* $\mathbf{x}_C$ is the set of variables belonging to clique $C$.
* $Z$ is the **Partition Function** (normalization constant, equivalent to the $Z$ in statistical physics).

---

### **Energy Form: The Gibbs Distribution**

The potential functions $\psi_C$ are often written in the exponential form, linking the distribution directly back to the principles of energy minimization (Chapter 9.1) and thermodynamics:

$$
\psi_C(\mathbf{x}_C) = e^{-E_C(\mathbf{x}_C)}
$$

Substituting this into the factorization yields the **Gibbs distribution**:

$$
p(\mathbf{x}) = \frac{1}{Z} e^{-E(\mathbf{x})}
$$
where $E(\mathbf{x}) = \sum_{C \in \mathcal{C}} E_C(\mathbf{x}_C)$ is the total system energy.

---

### **Physical Analogy: Spin Networks and Coupling Energy**

This exponential form makes the MRF the natural statistical generalization of foundational physical models:

* **Spin Systems:** MRFs are equivalent to **spin networks**. Each node acts as a particle/spin, and the potential functions $\psi_C$ encode the **coupling energy** between neighboring spins.
* **Ising and Potts Models:** A 2D lattice Ising model (Chapter 8) is a specific type of MRF where the cliques are pairs of nearest-neighbor sites, and the energy $E(\mathbf{x})$ is simply the Ising Hamiltonian.

Optimization techniques (like minimizing the energy $E(\mathbf{x})$ to find the ground state) and inference algorithms (like Belief Propagation) can thus be used interchangeably to solve problems in statistical physics and MRFs.

## **11.4 Conditional Independence and Markov Blankets**

In large, complex probabilistic graphical models (both directed BNs and undirected MRFs), the computational complexity of performing global inference appears daunting. However, the graph structure itself provides the solution by revealing inherent **conditional independencies** within the system, allowing for inference to be performed through **local computation**.

---

### **Conditional Independence: The Decoupling Principle**le**

Two variables, $A$ and $B$, are **conditionally independent** given a third set of variables, $C$, if knowing the state of $C$ provides all the information necessary to relate $A$ and $B$. Formally:

$$
p(A, B | C) = p(A | C) p(B | C)
$$

* **Implication:** The variables $A$ and $B$ are statistically **decoupled** once the state of $C$ is known. This is a profound simplifying principle, as it means the influence of distant parts of the network can be completely screened out by a local set of variables.

---

### **The Markov Blanket: The Informational Shell**

The concept of screening is formalized by the **Markov Blanket**. For any variable $x_i$ in the network, its Markov Blanket ($\text{MB}_i$) is the minimal set of surrounding variables that renders $x_i$ statistically independent of all other nodes in the graph.

* **Definition:** $x_i$ is conditionally independent of all other variables in the graph, given the state of its Markov Blanket.

The precise composition of the Markov Blanket differs slightly between Directed and Undirected Graphs:

| Graph Type | Markov Blanket Components |
| :--- | :--- |
| **Directed (BN)** | Parents, Children, and Co-Parents (the parents of $x_i$'s children). |
| **Undirected (MRF)** | Simply the set of **direct neighbors** of $x_i$. |

---

### **Analogy: Local Equilibrium and Screening**

* **Physical Analogy:** The Markov Blanket is analogous to achieving **local equilibrium** in a physical system. In a lattice model (like the Ising model, Section 11.3), the energy of a central spin $s_i$ depends only on the state of its nearest neighbors (its Markov Blanket). Distant spins influence $s_i$ only *through* the intermediate neighbors, whose influence is completely **screened out** by the local couplings.
* **Computational Implication:** Since local computations are sufficient for global inference, the size of the necessary calculation does not explode exponentially with the total number of variables $N$. This property is the foundation for the distributed computation necessary for **Belief Propagation** (Section 11.6).

## **11.5 Factor Graphs — Unified Representation**

While **Bayesian Networks (BNs)** and **Markov Random Fields (MRFs)** successfully capture conditional independence, their graphical factorization can differ slightly, particularly when dealing with shared factors between many variables. **Factor Graphs** offer a more explicit and unified representation that simplifies the mathematical description of the joint probability, which is particularly beneficial for inference algorithms.

---

### **Structure: The Bipartite Graph**

A Factor Graph is a **bipartite graph** consisting of two types of nodes:

1.  **Variable Nodes:** Represent the random variables $x_i$ (identical to nodes in BNs/MRFs).
2.  **Factor Nodes:** Represent the **local functions** or **potential functions** ($f_a$) that define the system's probability distribution.

An edge exists only between a variable node $x_i$ and a factor node $f_a$ if the variable $x_i$ is an argument of the function $f_a$.

---

### **Factorization: Explicit Product of Functions**

The joint probability distribution $p(\mathbf{x})$ is explicitly factored as a product of all factor functions $f_a$:

$$
p(\mathbf{x}) = \frac{1}{Z}\prod_a f_a(\mathbf{x}_a)
$$

* $\mathbf{x}_a$: The subset of variables connected to factor node $f_a$.
* $Z$: The Partition Function (normalization constant).

---

### **Benefit: Simplification of Message Passing**

Factor Graphs' bipartite structure is essential because it clearly separates the variables from the functions that couple them.

* **Clarity:** This representation makes it unambiguous which variables contribute to which potential, even in complex MRFs with large cliques.
* **Algorithmic Foundation:** The simplified, explicit factorization is the preferred starting point for iterative inference procedures like the **Belief Propagation (Sum–Product) algorithm** (Section 11.6). By operating on this simpler graph structure, the computational steps for transmitting "messages" (local probability information) are straightforward and easy to implement.

---

### **Analogy: Interaction Sites and Energy Landscape**

* **Variables $\leftrightarrow$ Particles:** The variable nodes represent the fundamental units of the system (particles, spins, or decisions).
* **Factors $\leftrightarrow$ Interaction Sites:** The factor nodes are analogous to **interaction sites** or local energy functions. They define the specific "forces" or coupling rules acting between the subset of particles they connect.

Together, the graph structure defines the geometry of the state space, and the factors define the **energy landscape** over which the system's probability distribution lives.

## **11.6 Belief Propagation (BP)**

**Belief Propagation (BP)**, also known as the **Sum–Product Algorithm**, is the core iterative inference procedure used in graphical models (especially Factor Graphs and tree-structured BNs/MRFs). It is the computational mechanism that allows the system to reach global statistical equilibrium through **distributed, local message passing**.

---

### **Goal and Mechanism: Collective Dynamics**

* **Goal:** The primary goal of BP is to efficiently compute the **marginal probability** $p(x_i)$ for every variable node $x_i$ in the network, conditioned on any evidence introduced. This marginal probability is the final, updated "belief" of the node.
* **Mechanism:** BP works by passing "messages" iteratively between neighboring nodes along the edges of the graph. Each message summarizes the sender's current belief about the receiver.

---

### **Message-Passing Equations**

In a Factor Graph formulation (Section 11.5), the BP algorithm involves two types of messages that pass back and forth until the entire system converges to a self-consistent state:

1.  **Variable-to-Factor Message ($m_{i \to a}$):** A variable node $x_i$ sends a message to its neighboring factor node $f_a$ summarizing the influence received from all its *other* neighbors.
2.  **Factor-to-Variable Message ($m_{a \to i}$):** A factor node $f_a$ sends a message back to its neighboring variable $x_i$, summarizing the probabilistic constraint imposed by the factor's function and the influence received from all other variables connected to $f_a$.

The general form of the message update is complex but embodies the idea of local accumulation:

$$
m_{i\to a}(x_i) = \prod_{k\in N(i)\setminus a} m_{k\to i}(x_i)
$$

This update ensures that every node updates its belief based only on the **local information** communicated by its direct neighbors.

---

### **Iteration and Convergence**

* **Iteration:** The process starts with uninformative messages (uniform priors) and iterates, with each node continually refining its belief based on the latest messages received from its neighbors.
* **Convergence:** For tree-structured graphs (graphs with no cycles/loops), BP is guaranteed to converge in a finite number of steps to the exact marginal probabilities. For graphs containing cycles ("loopy graphs"), BP is an approximate method but often still performs well (Section 11.8).

---

### **Physical Analogy: Distributed Relaxation**

Belief Propagation is the computational analogue of physical relaxation in a coupled system:

* **Statistical Equilibrium:** The iterative process of message passing is analogous to the system's **distributed relaxation to statistical equilibrium**.
* **Self-Consistency:** Each variable (node) acts like a local spin that adjusts its state until it achieves **collective consistency** with the forces (couplings/messages) exerted by its neighbors.
* **Mean-Field Connection:** For certain graphical models, the final BP beliefs are mathematically equivalent to the results obtained from a **mean-field approximation** in spin-glass theory. This demonstrates that inference algorithms are physically grounded in the dynamics of interacting systems.

## **11.7 Variational Inference and Free Energy Minimization**

The goal of Bayesian inference is to find the true **posterior distribution** $p(\mathbf{\theta}|\mathcal{D})$. However, for most large, complex graphical models, calculating this true posterior is analytically **intractable**. **Variational Inference (VI)** provides an optimization-based alternative that is deeply rooted in the concept of **Free Energy Minimization** from statistical physics.

---

### **Approximation Principle: The Trade-off**

Variational Inference replaces the intractable problem of computing the exact posterior $p$ with the tractable problem of **finding the best possible approximation $q$**.

* **Approximation $q$:** We define a parameterized, simpler distribution $q(\mathbf{\theta})$ (e.g., a simple Gaussian with independent factors) that we *can* compute.
* **Objective:** We seek to find the parameters of $q$ that minimize the **Kullback-Leibler (KL) divergence** (Section 2.2) between the approximation $q$ and the true posterior $p$:

$$
\min_q D_{\mathrm{KL}}(q||p)
$$

---

### **Defining the Variational Free Energy $\mathcal{F}(q)$**

By algebraically manipulating the definition of the KL divergence, the objective can be rewritten as maximizing the **Evidence Lower Bound (ELBO)** or, equivalently, minimizing the **Variational Free Energy functional, $\mathcal{F}(q)$**:

$$
\mathcal{F}(q) = \mathbb{E}_q[\ln q(\mathbf{\theta}) - \ln p(\mathcal{D}, \mathbf{\theta})]
$$

This minimization reveals that **statistical inference is recast as a familiar energy minimization problem**:
* **Minimizing $\mathcal{F}(q)$** ensures that the approximate distribution $q$ is pushed as close as possible to the true posterior $p$.
* The optimization goal is to find the distribution $q$ that causes the system to "relax" into a state of minimal informational free energy.

---

### **Physical Analogy: Mean-Field Approximation**

The VI framework is a direct and powerful analog of the **mean-field approximation** in statistical physics:

* **Mean-Field Theory:** Deals with complex, interacting systems (like spin glasses) by assuming that the interactions can be simplified. It approximates the true joint probability by a product of simpler, independent distributions.
* **Decoupling:** Variational inference often implements this by assuming the factors of $q$ are independent, effectively **decoupling** the parameters' joint probability. This strategy turns the global, coupled inference problem into a set of simpler, local optimization tasks.

The convergence of the optimization procedure (e.g., the EM algorithm used in GMMs, Chapter 3.8) to the minimum of $\mathcal{F}(q)$ is analogous to the physical system reaching a **minimal free energy configuration of beliefs**.

## **11.8 Loopy and Approximate Belief Propagation**

Inference using **Belief Propagation (BP)** (Section 11.6) is guaranteed to yield **exact marginal probabilities** only when applied to **tree-structured graphs** (graphs without cycles). However, many real-world systems, especially those in physics (like Ising models with nearest-neighbor interactions on a grid) and machine learning (like loopy error-correcting codes), are inherently structured as graphs with **cycles or loops**.

---

### **The Problem of Loops**

When BP algorithms are applied to a graph with loops, the messages passed around the cycle can recirculate indefinitely. A node receives a message, updates its belief, and sends a new message, which eventually returns to the node itself after traversing the loop. This leads to messages that are **statistically correlated**, violating the BP assumption that the messages summarize independent sub-trees.

---

### **Loopy Belief Propagation (LBP)**

Despite this mathematical violation, applying the same iterative BP rules to graphs with loops, known as **Loopy Belief Propagation (LBP)**, often proves to be an effective, highly scalable **approximation algorithm**.

* **Approximation:** For loopy graphs, LBP does **not** guarantee exact marginals, but it can often converge to a good approximation of them.
* **Convergence:** LBP is not guaranteed to converge, but when it does, the converged beliefs often correspond to a **stationary point** of the **Bethe approximation** to the Variational Free Energy (Section 11.7), suggesting a link to a form of relaxed equilibrium.

---

### **Interpretation: Long-Range Correlation**

* **Physical Interpretation:** Each loop in the graph introduces a potential for **long-range correlation** within the system. The complex pattern of convergence in LBP reflects the difficulty of precisely calculating the influence of these global, correlated forces. In a spin system (Chapter 11.3), loops are akin to **frustrated spins** (Chapter 8.10), where local energy requirements conflict, making the system difficult to solve.
* **Empirical Result:** LBP is widely used in applications like image denoising and error-correcting codes because it is computationally efficient and provides surprisingly good empirical performance, suggesting that the self-consistency achieved by the message-passing dynamics is often sufficient for practical inference.

## **11.9 Dynamic and Temporal Graphical Models**

The graphical models discussed so far (BNs and MRFs) primarily represent dependencies within a **static system**. However, many physical and informational systems, from molecule trajectories to signal processing, evolve over **time**. **Dynamic and Temporal Graphical Models** extend the standard framework to model these sequential processes, where the state at time $t$ depends on the state at time $t-1$.

---

### **The Dynamic Bayesian Network (DBN) Concept**

A **Dynamic Bayesian Network (DBN)** is not a single graph, but a sequence of identical graphs linked together across time steps. It assumes the **Markov property**: the future state ($t+1$) depends only on the current state ($t$) and not on the entire history of previous states ($t-1, t-2, \dots$).

The joint probability of a sequence is factored into two types of conditional probabilities:

1.  **Initial State Distribution:** $p(\mathbf{x}_1)$ (the starting condition).
2.  **Transition Probabilities:** $p(\mathbf{x}_t | \mathbf{x}_{t-1})$ (the dynamics, or how the system evolves).
3.  **Observation Probabilities:** $p(\mathbf{y}_t | \mathbf{x}_t)$ (how the true state $\mathbf{x}_t$ relates to the observed data $\mathbf{y}_t$).

---

### **Key Temporal Models**

Two canonical examples illustrate how DBNs are used to model sequential data:

* **Hidden Markov Model (HMM):** The HMM is the simplest and most famous DBN. It assumes that the true state of the system, $\mathbf{z}_t$, is **hidden** (unobserved), and we only observe a noisy manifestation, $\mathbf{x}_t$. The model factors the joint probability over the entire sequence of hidden states ($\mathbf{z}$) and observations ($\mathbf{x}$) as:

$$
p(\mathbf{x},\mathbf{z}) = p(z_1)\prod_t p(x_t|z_t)p(z_{t+1}|z_t)
$$
HMMs are widely used in sequence tagging, speech recognition, and modeling sequential physical data.

* **Kalman Filter:** This is the continuous-variable counterpart to the HMM, specifically tailored for systems where the hidden states and observations are governed by **linear relationships and Gaussian noise**. It provides an optimal method for state estimation (filtering), prediction, and smoothing of time series, used extensively in robotics and control systems.

---

### **Physical Analogy: System Evolution and State Estimation**

* **System Evolving in Time:** DBNs provide the framework for modeling the time evolution of a physical system. The transition probability $p(\mathbf{x}_t | \mathbf{x}_{t-1})$ encapsulates the system's **equations of motion** (its dynamics) in a probabilistic way.
* **Inference as State Estimation:** The core inference tasks—like finding the most likely sequence of hidden states given the observations (the Viterbi algorithm in HMMs)—are analogous to **state estimation** in physics or control theory. The system learns its internal dynamics while filtering out the noise from observations.

---

### **Bridge to Deep Learning**

Dynamic graphical models form the conceptual foundation for modern sequential neural networks. The structure of the HMM, which uses an internal hidden state ($\mathbf{z}_t$) coupled to the previous hidden state ($\mathbf{z}_{t-1}$), is the direct antecedent of the internal recurrence and memory mechanisms found in **Recurrent Neural Networks (RNNs)** (Chapter 13.5).

## **11.10 Worked Example — Belief Propagation on a Simple Graph**

To make the abstract concept of **Belief Propagation (BP)** (Section 11.6) concrete, we examine its application on the simplest type of graphical model: a linear chain. This example shows how **local, iterative updates** allow the system's beliefs to relax and converge to the true global statistical equilibrium.

---

### **Setup: A Binary Chain Model**

We consider a simple graphical model consisting of three binary random variables, $A$, $B$, and $C$ (each can be 0 or 1), connected in a chain:

$$
A - B - C
$$

* **Variables:** $\mathbf{x} = (A, B, C)$.
* **Goal:** Compute the **marginal probability** $p(B)$, which represents the final, updated belief of node $B$ after considering the influence of $A$ and $C$.
* **Potentials:** The joint distribution $p(A, B, C)$ is factored by local potential functions (analogous to the coupling energy of an Ising chain, Section 11.3):

$$
p(A, B, C) \propto \psi_{AB}(A, B) \cdot \psi_{BC}(B, C)
$$

---

### **The Procedure: Message Passing**

BP solves this problem by having $A$ send a message to $B$, and $C$ send a message to $B$, which $B$ then uses to compute its final belief.

1.  **Initialize Messages:** Since we have no prior evidence, the process starts with uniform (uninformative) messages.
2.  **$A \to B$ Message:** Node $A$ sends a message to $B$ summarizing its influence.
3.  **$C \to B$ Message:** Node $C$ sends a message to $B$ summarizing its influence.
4.  **Compute Marginal Belief:** Node $B$'s final belief, $p(B)$, is found by taking its initial local potential and multiplying it by all the messages it receives:

$$
\text{Belief}(B) \propto \psi_B(B) \cdot m_{A \to B}(B) \cdot m_{C \to B}(B)
$$

(Where $\psi_B(B)$ is the local potential of $B$, often uniform if no local evidence is given).

---

### **Interpretation: Ising Relaxation**

Because the graph is a **tree** (a simple chain has no cycles), BP is guaranteed to converge to the **exact marginal probability** in a single pass.

* **Local Updates $\rightarrow$ Global Inference:** The result demonstrates that even complex global dependencies are correctly handled by defining simple, local message passing rules.
* **Miniature Ising Relaxation:** The convergence of the belief to a stable value is analogous to a **miniature Ising relaxation**. Each node acts like a local spin that adjusts its state until it is aligned with the statistical "forces" (the messages) exerted by its neighbors, achieving a state of global statistical equilibrium.

## **11.11 Code Demo — Belief Propagation on a Binary Chain**

This demonstration implements the core **Belief Propagation (BP)** message-passing mechanism (Section 11.6) for the simple, three-node binary chain model ($A-B-C$) from the worked example (Section 11.10). The code simulates the iterative exchange of probability "messages" until the system's belief stabilizes, representing the relaxation to **statistical equilibrium**.

---

```python
import numpy as np

## --- 1. Define Pairwise Potentials (Couplings/Energy) ---

## Each matrix psi_XY(x, y) defines the unnormalized probability P(x,y).

## Indices: 0, 1 for the binary states (e.g., A=0, A=1)

## psi_AB[0, 0] = P(A=0, B=0) proportional to 3

## psi_AB[1, 0] = P(A=1, B=0) proportional to 1

psi_AB = np.array([[3,1],[1,3]])  # Coupling between A and B
psi_BC = np.array([[2,1],[1,2]])  # Coupling between B and C

## --- 2. Initialize Messages ---

## Messages start uniform (uninformative) or represent marginals of endpoints.

## The messages are 2-element vectors for the two binary states {0, 1}.

m_AtoB = np.ones(2)  # Message from A to B
m_CtoB = np.ones(2)  # Message from C to B

## --- 3. Iterative Message Passing (Inference) ---

## For a chain, the exact solution is found in one pass.

## We run multiple steps to demonstrate the convergence process.

for _ in range(10):
    # a) Compute Message from B to A (m_BtoA):
    # B receives influence from its *other* neighbor (C) and combines it with the A-B coupling.
    m_BtoA = psi_AB @ m_CtoB

    # b) Compute Message from B to C (m_BtoC):
    # B receives influence from its *other* neighbor (A) and combines it with the B-C coupling.
    m_BtoC = psi_BC @ m_AtoB

    # c) Update incoming messages for next iteration (Factor to Variable)
    # The message m_AtoB is updated using the *new* message m_BtoA and psi_AB.
    # Note: We use the transpose of psi_AB here for correct matrix multiplication logic.
    m_AtoB = psi_AB.T @ m_BtoA
    m_CtoB = psi_BC.T @ m_BtoC

    # Normalize messages to prevent numerical overflow (optional but good practice)
    m_AtoB /= np.sum(m_AtoB)
    m_CtoB /= np.sum(m_CtoB)

## --- 4. Compute Final Belief (Marginal Probability) ---

## The final belief for node B is proportional to the product of all messages received,

## multiplied by its own local potential (which is uniform here, so m_AtoB * m_CtoB).

belief_B = m_AtoB * m_CtoB
belief_B /= np.sum(belief_B)  # Final normalization to ensure sum(p)=1

print("Belief for node B (Final Marginal Probability):", belief_B)
```
**Sample Output:**
```python
Belief for node B (Final Marginal Probability): [0.5 0.5]
```

---

### **Interpretation**

  * **Relaxation Dynamics:** The loop (run `for _ in range(10):`) simulates the system iteratively reaching a state of **self-consistency**. The belief at node $B$ stabilizes when the messages it receives from $A$ and $C$ no longer change significantly.
  * **Local $\rightarrow$ Global:** The core operation $\text{Belief}(B) \propto m_{A \to B} \cdot m_{C \to B}$ shows that the complex global inference task is accomplished entirely by combining **local influences** (the messages).
  * **Ising Analogy:** The process is analogous to a **miniature Ising relaxation**. Each node adjusts its "spin" (its belief about its state) until it aligns with the local "coupling forces" defined by the potentials $\psi_{AB}$ and $\psi_{BC}$. The stable final belief represents the node's marginal probability at **statistical equilibrium**.

## **11.12 Graphical Models and Learning**

The structure of a graphical model (BN or MRF) defines how variables interact, allowing us to perform inference (Chapter 11.6). However, the specific parameters of the model—the conditional probability distributions (CPDs) or the potential functions ($\psi_C$)—must be determined, or **learned**, from the observed data. This transforms the task from pure statistical analysis into an optimization problem.

---

### **Parameter Learning: Inferring Interaction Strength**

**Parameter learning** involves adjusting the numerical values associated with the graph's fixed structure. This process is equivalent to finding the couplings ($J_{ij}$) and fields ($h_i$) that best explain the microstates observed in a physical system (Chapter 8.2).

* **Objective:** Maximize the **log-likelihood** of the observed data ($\mathcal{D}$) given the fixed graph structure:

$$
\max_{\mathbf{\theta}} \ln p(\mathcal{D}|\mathcal{G}, \mathbf{\theta})
$$
(Where $\mathbf{\theta}$ represents all the numerical parameters, such as the entries in the CPDs or the weights in the potential functions).
* **Method:** For fully observed data and simple graphs, maximizing likelihood can be done directly by counting frequencies. For complex or hidden variable models, iterative optimization is required, often using algorithms like **Expectation-Maximization (EM)**, which finds parameters that iteratively minimize the variational free energy (Chapter 9.6, 11.7).

!!! tip "Learning as Iterative Free Energy Minimization"
    Parameter learning in graphical models with hidden variables uses EM:
    
    * **E-step**: Compute expected sufficient statistics given current parameters (inference)
    * **M-step**: Update parameters to maximize expected log-likelihood (optimization)
    * **Convergence**: EM iteratively minimizes variational free energy $\mathcal{F}(q)$
    
    This unifies inference (computing posteriors) and learning (fitting parameters) as joint optimization.
    
---

### **Structure Learning: Discovering Connectivity**

A more challenging task is **structure learning**, where the goal is to infer the **topology** of the graph itself—determining which nodes are connected by edges. This is equivalent to discovering the **connectivity and interaction mechanism** in an unknown physical system.

* **Approach 1: Constraint-Based Methods:** These use statistical tests for **conditional independence** (Chapter 11.4) to determine which variables are decoupled given a set of others. If two variables are found to be independent, the edge between them is ruled out.
* **Approach 2: Score-Based Methods:** These define a quantitative **score** (e.g., maximizing the model evidence, $\ln p(\mathcal{D}|\mathcal{G})$) for every candidate graph $\mathcal{G}$. The problem then becomes a combinatorial optimization search over the vast space of possible graph structures to find the one with the highest score.

---

### **Analogy: Inferring Forces and Connectivity**

**Learning in graphical models is analogous to a physical system inferring both its forces and its structure**:

* **Parameter Learning $\leftrightarrow$ Inferring Force Strength:** The system knows its components are coupled (structure is fixed) but must infer the magnitude of the coupling energy ($J_{ij}$), analogous to determining the precise strength of the interaction potentials.
* **Structure Learning $\leftrightarrow$ Inferring Connectivity:** The system must determine which particles interact at all, analogous to inferring the physical constraints and bonds that define the system's topology.

Learning in graphical models is a complex form of **energy minimization over both the energy function's parameters and the underlying structural geometry**.

## **11.13 Applications Across Domains**

Probabilistic Graphical Models (PGMs), encompassing both Bayesian Networks (BNs) and Markov Random Fields (MRFs), provide a universal framework for understanding and modeling systems defined by **interdependencies**. The underlying mathematics of conditional probability and local factorization applies equally well across physics, engineering, and data science.

---

### **Unifying Concept: Graph as the Blueprint of Interaction**

In every domain, the graph serves as the **blueprint of interaction**, allowing algorithms to perform inference (e.g., estimating marginal probabilities or finding the most likely configuration) on complex, coupled systems.

| Domain | Graph Type | Variables & Dependencies | Role of PGM / Interpretation |
| :--- | :--- | :--- | :--- |
| **Physics** | MRF (Ising/Potts Fields) | Spins/sites and their local coupling energy $J_{ij}$. | **Statistical Mechanics:** Modeling correlation, finding the ground state (Chapter 8), and simulating disorder inference. |
| **Computer Vision** | MRF / Conditional Random Field (CRF) | Pixels, pixel values, and their neighbors. | **Image Segmentation/Denoising:** Enforcing smoothness and consistency by penalizing neighboring pixels from having vastly different labels (e.g., preventing a single white pixel in a black background). |
| **Genomics** | Bayesian Networks | Genes, proteins, and molecular activity levels. | **Gene Regulation Pathways:** Inferring the causal structure of molecular networks (e.g., $A \to B \to C$) and predicting how a change in one gene affects another. |
| **Natural Language Processing (NLP)** | HMM / CRF (Temporal Models) | Words, phrases, and their sequence in time. | **Sequence Tagging/Parsing:** Predicting the most likely sequence of hidden states (e.g., part-of-speech tags) given the observed words. |
| **Robotics & Control** | Dynamic Bayesian Networks | Sensor readings and internal state variables (HMM/Kalman Filter). | **Sensor Fusion/Localization:** Estimating the true state of a system (e.g., a robot's position) by filtering noisy, sequential sensor observations. |

The **unifying idea** is that any system whose **collective behavior is determined by local interactions** can be modeled using a probabilistic graph. The computational challenge shifts from solving complex differential equations to achieving statistical consistency via local message passing.

!!! example "MRF for Image Denoising"
    Apply MRF to denoise a corrupted binary image:
    
    * **Variables**: Each pixel $x_i \in \{0, 1\}$ (black or white)
    * **Observations**: Noisy pixel values $y_i$ (corrupted by flipping probability $p$)
    * **Pairwise potentials**: $\psi_{ij}(x_i, x_j) = e^{J}$ if $x_i = x_j$ (smoothness prior)
    * **Unary potentials**: $\psi_i(x_i) = e^{h \cdot \mathbb{1}_{x_i = y_i}}$ (observation likelihood)
    * **Inference**: Loopy BP computes marginals $p(x_i|\mathbf{y})$ to denoise image
    
    The MRF enforces spatial coherence—neighboring pixels prefer same values—while respecting observations.
    
## **11.14 Takeaways & Bridge to Part IV**

This chapter concluded the analytical section of **Part III: Learning as Inference**, demonstrating how probability theory can be translated into a structural, architectural framework. We showed that complex systems, defined by intricate local dependencies, can be rigorously modeled and solved by graphical methods.

---

### **Key Takeaways from Chapter 11**

* **Structure Unifies Science:** Probabilistic graphs unify **statistics, physics, and computation** through structure. The graph is the universal blueprint for modeling any system of interacting variables.
* **Inference as Dynamics:** The process of **Belief Propagation (BP)** (Section 11.6) transforms statistical inference into a set of **distributed relaxation dynamics**. The system's beliefs converge to **statistical equilibrium**, where local consistency (the Markov Blanket, Section 11.4) yields global truth.
* **Physics Analogy:** The same mathematics governs **spin alignment** in a disordered magnet (MRFs/Ising) as it does **causal reasoning** in a Bayesian Network. Inference is equivalent to **free-energy minimization** over the space of approximate probability distributions (Variational Inference, Section 11.7).
* **Learning is Structural:** The capacity to perform **structure learning** (Section 11.12) allows the system to infer the underlying forces and connections—the "laws of physics"—from the data alone.

??? question "Why Is Belief Propagation Exact Only on Trees?"
    BP assumes messages summarize independent sub-graphs (no information recirculation):
    
    * **Trees**: No cycles → each message path is unique → messages are truly independent
    * **Loopy graphs**: Cycles allow messages to recirculate → statistical correlations violate independence
    * **Physical analogy**: Loops create frustrated spins (conflicting local energies) like spin glasses
    * **Loopy BP**: Despite violation, often converges to good approximations (Bethe free energy)
    
    Trees guarantee exact marginals; loops require approximation but remain practical.
    
---

### **Bridge to Part IV: Deep Learning as Representation**

In **Part III: Learning as Inference** (Chapters 9–11), the probabilistic and graphical structures were largely **designed** or explicitly inferred from simple assumptions (e.g., linearity, shared covariance, predefined couplings).

In **Part IV: Deep Learning as Representation**, we take the final, massive leap: we introduce architectures that are powerful enough to **learn the entire structure, hierarchy, and features autonomously**.

* **From Fixed to Learned Structure:** We move from **handcrafted probability models** (like MRFs) to **massively parameterized neural networks**. The layers of a deep network (Chapter 12) become an *implicit* graphical model, automatically creating internal representations (latent variables) that simplify the probability landscape.
* **Learning the Transformation:** Instead of modeling $P(x_i|x_{j}, x_k)$, a deep network models the complex, non-linear function that transforms raw input $\mathbf{x}$ into a simple representation $\mathbf{z}$, over which inference becomes trivial (Chapter 13).

The next step is **Chapter 12: "The Perceptron and Neural Foundations,"** where we introduce the fundamental unit of deep learning—the neuron—and show how its basic optimization principle leads to complex, hierarchical architectures.

---

## **References**

[1] **Koller, D., & Friedman, N.** (2009). *Probabilistic Graphical Models: Principles and Techniques*. MIT Press. [Comprehensive textbook on BNs, MRFs, inference, and learning]
[2] **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer. [Chapter 8 on graphical models, variational inference, and EM algorithm]
[3] **Murphy, K. P.** (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press. [Modern treatment of PGMs with code examples]
[4] **Pearl, J.** (1988). *Probabilistic Reasoning in Intelligent Systems*. Morgan Kaufmann. [Foundational work on Bayesian Networks and belief propagation]
[5] **Wainwright, M. J., & Jordan, M. I.** (2008). "Graphical Models, Exponential Families, and Variational Inference." *Foundations and Trends in Machine Learning*. [Rigorous treatment of variational methods and exponential families]
[6] **Yedidia, J. S., Freeman, W. T., & Weiss, Y.** (2003). "Understanding Belief Propagation and Its Generalizations." *Exploring AI in the New Millennium*. [Loopy BP, Bethe free energy, and connections to statistical physics]
[7] **MacKay, D. J. C.** (2003). *Information Theory, Inference, and Learning Algorithms*. Cambridge University Press. [Information-theoretic perspective on inference and codes]
[8] **Mezard, M., & Montanari, A.** (2009). *Information, Physics, and Computation*. Oxford University Press. [Physics perspective on inference, spin glasses, and message passing]
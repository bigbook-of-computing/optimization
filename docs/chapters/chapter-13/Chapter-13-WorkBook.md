## Chapter 13: Hierarchical Representation Learning (Workbook)

The goal of this chapter is to study specialized **deep architectures** that embed geometric knowledge (space, time) into their structure, revealing how networks build a hierarchy of abstraction analogous to physical coarse-graining.

| Section | Topic Summary |
| :--- | :--- |
| **13.1** | From Perceptrons to Hierarchies |
| **13.2** | The Geometry of Deep Representations |
| **13.3** | Convolutional Neural Networks (CNNs) — Spatial Structure |
| **13.4** | Hierarchical Feature Maps |
| **13.5** | Temporal Hierarchies — Recurrent Neural Networks (RNNs) |
| **13.6** | Autoencoders — Learning Internal Coordinates |
| **13.7** | Variational Autoencoders (VAEs) — Probabilistic Extension |
| **13.8** | Spatial–Temporal Hybrids |
| **13.9** | Regularization and Generalization in Deep Networks |
| **13.10–13.15**| Visualization, Worked Example, and Takeaways |

---

### 13.1 From Perceptrons to Hierarchies

> **Summary:** Deep networks achieve computational power by stacking the non-linear unit (perceptron), where **composition of non-linear mappings** yields immense expressive power. The sequential transformation (layer $l \to l+1$) is analogous to the **Renormalization Group (RG) framework** in physics, as each layer performs a step of **coarse-graining** to extract effective, abstract variables.

#### Quiz Questions

**1. The Universal Approximation Theorem proves that a single hidden layer (of sufficient size) and non-linear activation can approximate any continuous function. The added benefit of *stacking* layers (depth) is to facilitate:**

* **A.** The loss of all variance.
* **B.** **The hierarchical extraction of emergent features**. (**Correct**)
* **C.** The reduction of the learning rate $\eta$.
* **D.** The calculation of the geodesic distance.

**2. The process where a deep network iteratively removes microscopic details and redundancy to encode a new, effective state variable $\mathbf{h}^{(l+1)}$ is directly analogous to which physical framework?**

* **A.** The Ising model Hamiltonian.
* **B.** **The Renormalization Group (RG) framework**. (**Correct**)
* **C.** Quantum tunneling.
* **D.** Orthogonal projection.

---

#### Interview-Style Question

**Question:** Explain the philosophical significance of viewing each layer of a deep neural network as a step of **coarse-graining** in the Renormalization Group sense?

**Answer Strategy:** The significance is that it provides a physical justification for the network's structure. In physics, coarse-graining is necessary to reveal macroscopic laws by ignoring high-frequency, microscopic noise. By performing this iteratively, the network is automatically learning a **hierarchical theory of the data**; it progresses from raw input to abstract **order parameters** that genuinely describe the system's macroscopic state, without any human defining those intermediate features.

---
***

### 13.2 The Geometry of Deep Representations

> **Summary:** Deep networks manipulate the **geometry of the data space** by performing successive **change of coordinates**. The primary goal is to **unfold, disentangle, and flatten** the complex, high-dimensional data manifold ($\mathcal{M}$). The network seeks a final, deep representation where data belonging to different classes are trivially **linearly separable**. This hierarchical process is analogous to **progressively flattening a rough potential surface** for easier optimization dynamics.

#### Quiz Questions

**1. The primary geometric goal of the successive transformations performed by the hidden layers in a deep network is to:**

* **A.** Preserve the Euclidean distance between all points.
* **B.** **Disentangle the structure and flatten the manifold for linear separation**. (**Correct**)
* **C.** Increase the overall dimension $D$.
* **D.** Maximize the total energy.

**2. In the physical analogy, the process of hierarchical learning is compared to applying successive transformations that gradually flatten a curved potential surface. The final state of this flattening allows the solution to be found by:**

* **A.** Introducing a high learning rate $\eta$.
* **B.** **A simple linear classifier**. (**Correct**)
* **C.** Quantum annealing.
* **D.** Calculating the partition function.

---

#### Interview-Style Question

**Question:** How does the design goal of a deep network (to flatten the data manifold) relate to the concept of the final output layer being a simple **linear classifier** (Chapter 10)?

**Answer Strategy:** The high-level layers of a deep network are doing the hard work: they are warping the space through non-linearity until the data, which was highly complex in the raw input space, becomes **linearly separable** in the final hidden feature space. Once the data is linearly separable, the final classification layer only needs to apply a simple linear model (like Logistic Regression or a Perceptron) to draw a straight line or plane to separate the classes, making the final classification step computationally trivial.

---
***

### 13.3 Convolutional Neural Networks (CNNs) — Spatial Structure

> **Summary:** CNNs are specialized architectures that embed knowledge of **spatial structure**. The core **convolution layer** replaces dense weights with a small, shared **kernel**. This design incorporates two **inductive biases**: **Locality** (small receptive fields mimic nearest-neighbor interaction) and **Weight Sharing** (enforcing translation invariance). **Pooling** layers perform the coarse-graining step. The entire process is analogous to **real-space renormalization** in physics.

#### Quiz Questions

**1. The CNN feature of **Weight Sharing** is a computational mechanism that directly enforces the physical property of:**

* **A.** Entropy reduction.
* **B.** **Translation symmetry (invariance)**. (**Correct**)
* **C.** Temporal recurrence.
* **D.** Orthogonal projection.

**2. In the CNN analogy to physics, the small, local **kernel** in the convolution layer models which structural feature?**

* **A.** Global correlation.
* **B.** **Nearest-neighbor interaction**. (**Correct**)
* **C.** The bias-variance trade-off.
* **D.** The time step $\Delta t$.

---

#### Interview-Style Question

**Question:** The CNN uses both **Convolution** and **Pooling** layers. Explain the respective function of each layer in terms of the **Renormalization Group (RG)** framework.

**Answer Strategy:**
1.  **Convolution $\leftrightarrow$ Feature Extraction:** The convolution layer applies local filters to extract fundamental patterns (e.g., edges, textures) from the neighborhood, identifying **short-range correlations**.
2.  **Pooling $\leftrightarrow$ Coarse-Graining:** The pooling layer summarizes and reduces the spatial dimension (e.g., taking the maximum activation over a $2 \times 2$ block). This step is the direct analogue of **coarse-graining** in RG theory: removing high-frequency, microscopic noise and redundancy to reveal the simplified, lower-dimensional structure.

---
***

### 13.4 Hierarchical Feature Maps

> **Summary:** Deep CNNs build a hierarchy of features, progressing from **low-level features** (microscopic: edges, lines) to **high-level features** (macroscopic: semantic objects, order parameters). This progression mirrors the **hierarchy of excitations** in physical systems. The most active neurons in deep layers often act as **emergent order parameters**, which are abstract variables that characterize the system's macroscopic state.

### 13.5 Temporal Hierarchies — Recurrent Neural Networks (RNNs)

> **Summary:** **Recurrent Neural Networks (RNNs)** are specialized architectures for **sequential data** (time series, language). They achieve **memory** by introducing **feedback**, where the hidden state $\mathbf{h}_t$ depends on both the current input $\mathbf{x}_t$ and the previous state $\mathbf{h}_{t-1}$. The recurrent weight matrix ($W_h$) defines the **transition function** of this non-linear **dynamical system**. **LSTMs** and **GRUs** use internal gating mechanisms to solve the **vanishing gradient problem** and maintain long-term memory.

### 13.6 Autoencoders — Learning Internal Coordinates

> **Summary:** **Autoencoders (AEs)** are architectures for **unsupervised learning** and dimensionality reduction. The AE uses an **Encoder** to map high-D input to a low-D **latent code ($\mathbf{z}$)** and a **Decoder** to reconstruct the input. The objective is to minimize **reconstruction loss**, forcing $\mathbf{z}$ to become the optimal **information compressor**. The entire AE operation is analogous to modeling the **Renormalization Group (RG) flow** forward (encoder) and backward (decoder).

### 13.7 Variational Autoencoders (VAEs) — Probabilistic Extension

> **Summary:** **Variational Autoencoders (VAEs)** embed **probabilistic inference** into the AE structure. VAEs train by maximizing the **Evidence Lower Bound (ELBO)**, which explicitly balances two terms: **Reconstruction Loss (Energy)** and the **KL Divergence** (forcing the latent space to adhere to a regular prior, analogous to minimizing **Entropy Cost**). The VAE's objective function directly embodies the thermodynamic **Energy–Entropy duality**.

### 13.8 Spatial–Temporal Hybrids

> **Summary:** **ConvRNNs** (Convolutional RNNs) combine CNN kernels (for spatial locality) and recurrent connections (for temporal memory) to model systems coupled across both space and time (e.g., fluid dynamics, video prediction). These models act as **probabilistic numerical solvers** that learn the **transition function** (equations of motion) of a continuous physical field.

### 13.9 Regularization and Generalization in Deep Networks

> **Summary:** Deep networks use **regularization** to control complexity and improve **generalization** by managing the **Bias–Variance Tradeoff**. **L2 Weight Decay** acts as a Gaussian prior (energetic constraint). **Dropout** acts as **entropy injection**, forcing the network to average over an ensemble of thinned sub-networks. **Early Stopping** acts as **cooling control**, halting training at a point of optimal thermal balance.

---

## 💡 Hands-On Project Ideas 🛠️

These projects focus on implementing and testing the specialized architectures and their physical analogies.

### Project 1: Simulating RNN Memory and Dynamics

* **Goal:** Implement a simple RNN and demonstrate its ability to maintain memory over a short sequence.
* **Setup:** Define a simple, small RNN with a single recurrent layer (Section 13.5).
* **Steps:**
    1.  Train the RNN on a simple sequential prediction task (e.g., predicting the next item in a repeating sequence like `1, 0, 1, 0, ...`).
    2.  Test the trained network by feeding it a sequence with a gap (e.g., `1, 0, 1, [Gap], 1`).
* ***Goal***: Show that the hidden state $\mathbf{h}_t$ successfully carries the necessary information (memory) across the temporal gap to correctly predict the missing element, illustrating its function as a dynamical system.

### Project 2: Autoencoder Manifold Discovery (Non-Linear PCA)

* **Goal:** Use an Autoencoder (AE) to discover the non-linear manifold of a simple dataset, contrasting it with PCA.
* **Setup:** Generate a 2D **curved** dataset (e.g., a simple spiral or C-shaped manifold in $\mathbb{R}^2$).
* **Steps:**
    1.  Design a simple AE with an Encoder and Decoder (MLPs with ReLU) and a 1D latent code ($\mathbf{z} \in \mathbb{R}^1$).
    2.  Train the AE to minimize reconstruction loss.
    3.  Extract the 1D latent code $\mathbf{z}$ for all samples and plot the raw data colored by the value of $\mathbf{z}$.
* ***Goal***: Show that the AE learns a continuous, single coordinate that successfully "unrolls" the curved structure, performing a **non-linear dimensionality reduction** superior to linear PCA.

### Project 3: Quantifying the VAE's Energy–Entropy Balance

* **Goal:** Implement the VAE objective and analyze the trade-off between its two loss terms (Energy/Reconstruction vs. Entropy/KL).
* **Setup:** Design a simple VAE and train it on a small dataset (e.g., MNIST digits).
* **Steps:**
    1.  Track and plot the value of the **Reconstruction Loss** (the energy term) and the **KL Divergence** (the entropy cost) separately over the training epochs.
    2.  Introduce a large weighting factor ($\beta \gg 1$) on the KL term (a "$\\beta$-VAE").
* ***Goal***: Show that a large $\beta$ forces the KL term down (low entropy cost) but significantly *increases* the Reconstruction Loss (poor image quality), demonstrating the explicit **energy–entropy trade-off** governed by the $\beta$ parameter.

### Project 4: Simulating Coarse-Graining in a CNN (Feature Map)

* **Goal:** Visually demonstrate the **hierarchical abstraction** of a CNN (Section 13.4).
* **Setup:** Use a pre-trained simple CNN model (like the one in the demo) on an image dataset (e.g., MNIST).
* **Steps:**
    1.  Select a single input image (e.g., the digit '4').
    2.  Extract and visualize the feature maps after the **first convolutional layer** (low level).
    3.  Extract and visualize the feature maps after the **final pooling layer** (high level).
* ***Goal***: Show that the first layer's maps highlight simple, raw edges (microscopic), while the final layer's maps are highly compressed and abstract, showing only the presence of the full object (macroscopic), confirming the **RG coarse-graining principle**.

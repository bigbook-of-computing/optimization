
# **Chapter-13: Quizes**

---

!!! note "Quiz"
    **1. What is the primary advantage of using a deep, hierarchical architecture (like a CNN) over a "flat" model (like a single-layer perceptron) for structured data?**

    - A. Deep models have fewer parameters.
    - B. Deep models can learn a hierarchy of features, where simple patterns are composed into complex, abstract concepts.
    - C. Deep models are guaranteed to find a global minimum.
    - D. Deep models do not require non-linear activation functions.

    ??? info "See Answer"
        **Correct: B**

        *(Depth allows the network to build a compositional model of the data, mirroring the hierarchical structure of the physical world. This is analogous to the Renormalization Group (RG) framework.)*

---

!!! note "Quiz"
    **2. The process of a deep network transforming data through its layers to make it linearly separable in the final hidden space is best described as:**

    - A. Manifold compression.
    - B. Manifold unfolding and flattening.
    - C. Manifold rotation.
    - D. Manifold interpolation.

    ??? info "See Answer"
        **Correct: B**

        *(Deep networks learn a sequence of coordinate changes that progressively disentangle and flatten the curved data manifold, making the final classification task trivial for a linear model.)*

---

!!! note "Quiz"
    **3. What two core inductive biases are embedded in a Convolutional Neural Network (CNN) to efficiently process spatial data like images?**

    - A. Global correlation and temporal recurrence.
    - B. Non-linearity and high dimensionality.
    - C. Local correlation (locality) and translation invariance (weight sharing).
    - D. Stochasticity and memory.

    ??? info "See Answer"
        **Correct: C**

        *(CNNs assume that nearby pixels are strongly correlated (locality, handled by small kernels) and that a feature is the same regardless of its position (translation invariance, handled by sharing weights).)*

---

!!! note "Quiz"
    **4. In a CNN, what is the direct physical analogy for the **pooling** layer (e.g., MaxPooling)?**

    - A. Applying a local field operator.
    - B. A change of basis.
    - C. Coarse-graining or down-sampling.
    - D. A non-linear activation.

    ??? info "See Answer"
        **Correct: C**

        *(The pooling layer reduces the spatial dimension by summarizing a local region, which is a direct implementation of the coarse-graining step in the Renormalization Group (RG) framework.)*

---

!!! note "Quiz"
    **5. When visualizing the filters learned by a deep CNN, what kind of patterns do the filters in the very first convolutional layer typically represent?**

    - A. Complex objects like faces or cars.
    - B. Random noise.
    - C. Simple, generic features like edges, corners, and color blobs.
    - D. The final classification labels.

    ??? info "See Answer"
        **Correct: C**

        *(The earliest layers learn the most fundamental, microscopic building blocks of the image, which are then composed into more complex shapes by deeper layers.)*

---

!!! note "Quiz"
    **6. A Recurrent Neural Network (RNN) is specifically designed to model what kind of data structure?**

    - A. Data with strong spatial locality.
    - B. Data that lies on a low-dimensional manifold.
    - C. Sequential data with temporal dependencies.
    - D. Data that is perfectly independent and identically distributed (i.i.d.).

    ??? info "See Answer"
        **Correct: C**

        *(RNNs use feedback loops to maintain a "memory" or hidden state that evolves over time, making them ideal for modeling sequences like language, time series, or physical trajectories.)*

---

!!! note "Quiz"
    **7. The core mathematical operation of an RNN, $\mathbf{h}_t = \phi(W_h \mathbf{h}_{t-1} + W_x \mathbf{x}_t + \mathbf{b})$, defines it as what kind of physical system?**

    - A. A static equilibrium system.
    - B. A non-linear dynamical system.
    - C. A system at zero temperature.
    - D. An isolated quantum system.

    ??? info "See Answer"
        **Correct: B**

        *(The recurrent update rule defines how the system's state ($\mathbf{h}_t$) evolves based on its previous state and current input, which is the definition of a dynamical system.)*

---

!!! note "Quiz"
    **8. What is the primary purpose of an Autoencoder (AE) architecture?**

    - A. To classify data into one of K classes.
    - B. To perform unsupervised learning by learning an efficient, compressed latent representation of the data.
    - C. To model the temporal evolution of a system.
    - D. To add noise to the input data for regularization.

    ??? info "See Answer"
        **Correct: B**

        *(An AE is trained to reconstruct its own input after passing it through a low-dimensional "bottleneck," forcing it to learn the most salient features or "internal coordinates" of the data.)*

---

!!! note "Quiz"
    **9. In an Autoencoder, the encoder maps the input $\mathbf{x}$ to the latent code $\mathbf{z}$, and the decoder maps $\mathbf{z}$ back to a reconstruction $\mathbf{x}'$. This process is analogous to:**

    - A. A forward and backward pass of backpropagation.
    - B. A Fourier and inverse Fourier transform.
    - C. A forward (coarse-graining) and backward (reconstruction) flow in the Renormalization Group.
    - D. A query and response from a database.

    ??? info "See Answer"
        **Correct: C**

        *(The encoder compresses information (coarse-grains), while the decoder attempts to reconstruct it, forcing the latent code to capture the essential macroscopic variables.)*

---

!!! note "Quiz"
    **10. What is the key difference between a standard Autoencoder (AE) and a Variational Autoencoder (VAE)?**

    - A. VAEs can only be used for supervised learning.
    - B. VAEs are deterministic, while AEs are probabilistic.
    - C. VAEs introduce a probabilistic framework, forcing the latent space to follow a prior distribution (e.g., Gaussian) through a KL divergence term in the loss.
    - D. VAEs do not have a decoder.

    ??? info "See Answer"
        **Correct: C**

        *(VAEs are generative models that learn a structured, probabilistic latent space, allowing for meaningful generation of new samples by drawing from that space.)*

---

!!! note "Quiz"
    **11. The loss function of a VAE, the Evidence Lower Bound (ELBO), represents a trade-off between what two quantities?**

    - A. Bias and variance.
    - B. Reconstruction accuracy (Energy) and latent space regularity (Entropy).
    - C. Learning rate and momentum.
    - D. Spatial and temporal resolution.

    ??? info "See Answer"
        **Correct: B**

        *(The ELBO balances the need to accurately reconstruct the data (lowering potential energy) with the need to keep the latent distribution simple and regular (lowering entropy cost via the KL term).)*

---

!!! note "Quiz"
    **12. A ConvRNN, which combines convolutional operations within a recurrent loop, is best suited for what type of task?**

    - A. Classifying static images.
    - B. Modeling the evolution of spatiotemporal fields, like in video prediction or weather forecasting.
    - C. Compressing unstructured tabular data.
    - D. Generating text.

    ??? info "See Answer"
        **Correct: B**

        *(ConvRNNs are designed to handle data that has both spatial structure at each time step and temporal dependencies between time steps.)*

---

!!! note "Quiz"
    **13. The regularization technique of **Dropout** is physically analogous to:**

    - A. Cooling a system to its ground state.
    - B. Applying a strong external magnetic field.
    - C. Severe entropy injection, forcing the system to find robust, redundant pathways.
    - D. Measuring a quantum system, causing wavefunction collapse.

    ??? info "See Answer"
        **Correct: C**

        *(By randomly deactivating neurons, Dropout prevents overfitting by making it impossible for the network to rely on any single pathway, analogous to increasing the disorder/entropy to find a more robust solution.)*

---

!!! note "Quiz"
    **14. What is the purpose of using a visualization technique like a **saliency map**?**

    - A. To show the distribution of weights in the final layer.
    - B. To project the high-dimensional latent space into 2D.
    - C. To identify which input features (e.g., pixels) were most influential for a given prediction.
    - D. To plot the training and validation loss curves.

    ??? info "See Answer"
        **Correct: C**

        *(Saliency maps are computed from the gradient of the output with respect to the input, highlighting the areas the network "paid attention to" for its decision.)*

---

!!! note "Quiz"
    **15. If you project the hidden states of a network trained on physics simulation data (e.g., spin configurations) into 2D using t-SNE, what do the resulting clusters often correspond to?**

    - A. The learning rate schedule.
    - B. The different activation functions used.
    - C. The distinct thermodynamic phases or metastable states of the physical system.
    - D. The number of parameters in the model.

    ??? info "See Answer"
        **Correct: C**

        *(This confirms that the network has learned to identify the macroscopic order parameters that distinguish the system's different phases, effectively creating a learned phase diagram.)*

---

!!! note "Quiz"
    **16. Why does a CNN typically outperform a fully-connected MLP on an image classification task, even with fewer parameters?**

    - A. Because CNNs use a more advanced optimization algorithm.
    - B. Because the architectural priors of a CNN (locality, weight sharing) match the intrinsic structure of images, providing powerful regularization.
    - C. Because MLPs cannot use the ReLU activation function.
    - D. Because CNNs are always deeper than MLPs.

    ??? info "See Answer"
        **Correct: B**

        *(The CNN's inductive biases are a form of built-in knowledge about the problem domain, which makes learning far more efficient and effective than for a generic MLP that must learn these properties from scratch.)*

---

!!! note "Quiz"
    **17. The "receptive field" of a neuron in a convolutional layer refers to:**

    - A. The entire input image.
    - B. The set of all neurons in the previous layer.
    - C. The small, local region of the input that the neuron is connected to.
    - D. The range of values the neuron's activation can take.

    ??? info "See Answer"
        **Correct: C**

        *(The receptive field is the local window defined by the kernel size, which enforces the principle of locality.)*

---

!!! note "Quiz"
    **18. What is the primary challenge that LSTM and GRU architectures were designed to solve in standard RNNs?**

    - A. The exploding gradient problem.
    - B. The inability to process sequential data.
    - C. The vanishing gradient problem, which hinders learning of long-term dependencies.
    - D. The high computational cost of the activation function.

    ??? info "See Answer"
        **Correct: C**

        *(The gating mechanisms in LSTMs and GRUs allow the network to selectively retain or forget information over long time scales, preventing the error signal from decaying to zero.)*

---

!!! note "Quiz"
    **19. The Transformer architecture differs from CNNs and RNNs by using what mechanism to model dependencies?**

    - A. Recurrent feedback loops.
    - B. Local convolutional kernels.
    - C. A self-attention mechanism that computes dynamic, global correlations between all elements.
    - D. A fixed, predefined graph structure.

    ??? info "See Answer"
        **Correct: C**

        *(Self-attention allows the model to weigh the importance of all other elements in the input when processing a given element, capturing complex, non-local interactions that CNNs and RNNs might miss.)*

---

!!! note "Quiz"
    **20. The regularization technique of **Early Stopping** is physically analogous to:**

    - A. A phase transition.
    - B. Controlled cooling, stopping the optimization at a point of optimal thermal balance to prevent "freezing" into a poor, overfit minimum.
    - C. Increasing the system's energy.
    - D. Applying an external force.

    ??? info "See Answer"
        **Correct: B**

        *(Early stopping prevents the model from perfectly memorizing the training data by halting the process when generalization performance (on a validation set) is at its peak.)*

---

!!! note "Quiz"
    **21. In the context of deep learning, what is the "Manifold Hypothesis"?**

    - A. The hypothesis that all data can be modeled by a Gaussian distribution.
    - B. The hypothesis that real-world high-dimensional data actually lies on or near a low-dimensional, curved manifold.
    - C. The hypothesis that deeper networks are always better.
    - D. The hypothesis that the loss landscape is always convex.

    ??? info "See Answer"
        **Correct: B**

        *(This hypothesis motivates the use of deep learning to learn the non-linear transformations required to "unfold" this manifold and make the data linearly separable.)*

---

!!! note "Quiz"
    **22. The final dense layers of a CNN act as:**

    - A. The primary feature extractors for local patterns.
    - B. A classifier that operates on the high-level, abstract features produced by the convolutional and pooling layers.
    - C. A mechanism for down-sampling the input image.
    - D. A regularization technique to prevent overfitting.

    ??? info "See Answer"
        **Correct: B**

        *(After the convolutional base has extracted a rich set of abstract features, the final dense layers perform the classification task based on this high-level representation.)*

---

!!! note "Quiz"
    **23. If you train a VAE with a very high $\beta$ value (in a $\beta$-VAE), what is the likely outcome?**

    - A. The reconstructions will be perfect, but the latent space will be disorganized.
    - B. The latent space will be very smooth and close to the Gaussian prior, but the reconstructions will be blurry or of poor quality.
    - C. The training will be much faster.
    - D. The model will be immune to overfitting.

    ??? info "See Answer"
        **Correct: B**

        *(A high $\beta$ places a strong penalty on the KL divergence (entropy cost), forcing the latent space to be simple at the expense of reconstruction accuracy (energy).)*

---

!!! note "Quiz"
    **24. The overall view of a deep network as a physical system involves seeing the weights as couplings, gradients as forces, and learning as:**

    - A. A process of energy conservation.
    - B. A random walk in parameter space.
    - C. A process of stochastic relaxation or energy dissipation.
    - D. A reversible thermodynamic cycle.

    ??? info "See Answer"
        **Correct: C**

        *(The optimization algorithm (like SGD) acts to dissipate the system's potential energy (the loss) until it settles into a low-energy equilibrium state.)*

---

!!! note "Quiz"
    **25. This chapter focuses on discriminative models ($P(y|\mathbf{x})$). Chapter 14 will shift focus to what kind of models?**

    - A. Simpler linear models.
    - B. Generative models, which learn the full data distribution $P(\mathbf{x})$ and can synthesize new samples.
    - C. Non-parametric models.
    - D. Reinforcement learning models.

    ??? info "See Answer"
        **Correct: B**

        *(The bridge to Chapter 14 is the transition from recognizing patterns to generating them by explicitly modeling the energy landscape of the data distribution.)*

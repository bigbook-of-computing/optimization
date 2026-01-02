
# **Chapter-14: Quizes**

---


!!! note "Quiz"
**1. What is the fundamental shift in the learning objective when moving from discriminative to generative modeling?**
*   A. From learning a regression function to a classification boundary.
*   B. From learning a conditional probability distribution $P(y|\mathbf{x})$ to a joint probability distribution $P(\mathbf{x})$.
*   C. From minimizing a loss function to maximizing a reward signal.
*   D. From using supervised data to using unsupervised data.

??? info "See Answer"
    **B. From learning a conditional probability distribution $P(y|\mathbf{x})$ to a joint probability distribution $P(\mathbf{x})$.**
    
    **Explanation:** Generative modeling's core task is to learn the underlying structure of the data itself, represented by the joint distribution $P(\mathbf{x})$, enabling it to create new samples. Discriminative models only learn the boundary between classes, $P(y|\mathbf{x})$.

---

!!! note "Quiz"
**2. In the Energy-Based Model (EBM) framework, the probability of a configuration $\mathbf{x}$ is related to its energy $E_\theta(\mathbf{x})$ by which formula?**
*   A. $p(\mathbf{x}) = E_\theta(\mathbf{x}) / Z_\theta$
*   B. $p(\mathbf{x}) = \log(E_\theta(\mathbf{x})) / Z_\theta$
*   C. $p(\mathbf{x}) = e^{E_\theta(\mathbf{x})} / Z_\theta$
*   D. $p(\mathbf{x}) = e^{-E_\theta(\mathbf{x})} / Z_\theta$

??? info "See Answer"
    **D. $p(\mathbf{x}) = e^{-E_\theta(\mathbf{x})} / Z_\theta$**
    
    **Explanation:** This is the Boltzmann distribution, which defines the energy-probability duality. High-probability states correspond to low-energy configurations.

---

!!! note "Quiz"
**3. The training process of an EBM is often described as "energy landscape sculpting." What does this metaphor imply?**
*   A. The model physically carves a silicon chip to store weights.
*   B. The model adjusts its parameters to create low-energy "potential wells" around data points and high-energy "barriers" in empty regions.
*   C. The model smooths out the loss function to avoid all local minima.
*   D. The model learns to generate images of landscapes.

??? info "See Answer"
    **B. The model adjusts its parameters to create low-energy "potential wells" around data points and high-energy "barriers" in empty regions.**
    
    **Explanation:** Learning in an EBM is equivalent to shaping the energy function so that its minima correspond to the locations of observed data in the state space.

---

!!! note "Quiz"
**4. The gradient of the log-probability for an EBM, $\nabla_\theta \log p_\theta(\mathbf{x})$, consists of a "Data Term" and a "Model Term." What is the role of the Model Term?**
*   A. To decrease the energy of the observed data samples.
*   B. To calculate the partition function exactly.
*   C. To increase the energy of samples drawn from the model's own distribution, pushing up the energy in "non-data" regions.
*   D. To ensure the weights of the network remain small.

??? info "See Answer"
    **C. To increase the energy of samples drawn from the model's own distribution, pushing up the energy in "non-data" regions.**
    
    **Explanation:** The Model Term, $\mathbb{E}_{p_\theta}[\nabla_\theta E_\theta(\mathbf{x})]$, acts as a repulsive force, preventing the energy landscape from collapsing to a single point and ensuring that regions without data have high energy.

---

!!! note "Quiz"
**5. Why is the partition function $Z_\theta$ a major challenge in training most Energy-Based Models?**
*   A. It is always equal to zero.
*   B. It is an imaginary number that cannot be represented by standard data types.
*   C. It requires integrating or summing over an exponentially large or infinite state space, making it computationally intractable.
*   D. It can only be calculated if the data is perfectly noise-free.

??? info "See Answer"
    **C. It requires integrating or summing over an exponentially large or infinite state space, making it computationally intractable.**
    
    **Explanation:** This intractability is the central problem in statistical mechanics and EBMs, forcing the use of sampling-based approximations like MCMC.

---

!!! note "Quiz"
**6. What is the key architectural difference between a general Boltzmann Machine (BM) and a Restricted Boltzmann Machine (RBM)?**
*   A. RBMs use continuous units, while BMs use binary units.
*   B. RBMs have connections between hidden units, but BMs do not.
*   C. RBMs have a bipartite graph structure with no intra-layer (visible-visible or hidden-hidden) connections.
*   D. RBMs can only be used for discriminative tasks.

??? info "See Answer"
    **C. RBMs have a bipartite graph structure with no intra-layer (visible-visible or hidden-hidden) connections.**
    
    **Explanation:** This restriction makes the conditional probabilities $P(\mathbf{h}|\mathbf{v})$ and $P(\mathbf{v}|\mathbf{h})$ factorize, allowing for efficient inference and sampling.

---

!!! note "Quiz"
**7. The Contrastive Divergence (CD) algorithm is used to train RBMs. What is the core idea behind CD?**
*   A. It computes the exact gradient of the log-likelihood.
*   B. It approximates the "Model Term" of the gradient using a short MCMC chain, contrasting data statistics with model statistics.
*   C. It uses backpropagation to minimize a reconstruction error.
*   D. It trains a separate discriminator network to guide the RBM.

??? info "See Answer"
    **B. It approximates the "Model Term" of the gradient using a short MCMC chain, contrasting data statistics with model statistics.**
    
    **Explanation:** CD provides a fast but biased estimate of the gradient by running only a few steps of Gibbs sampling, making RBM training practical.

---

!!! note "Quiz"
**8. How does a Variational Autoencoder (VAE) differ from a standard Autoencoder (AE)?**
*   A. A VAE has no decoder, only an encoder.
*   B. A VAE's encoder outputs a single, deterministic latent vector.
*   C. A VAE's encoder outputs parameters (mean and variance) for a probability distribution over the latent space, introducing stochasticity.
*   D. A VAE can only reconstruct binary data.

??? info "See Answer"
    **C. A VAE's encoder outputs parameters (mean and variance) for a probability distribution over the latent space, introducing stochasticity.**
    
    **Explanation:** This probabilistic latent space is the key innovation of the VAE, allowing it to function as a generative model by sampling from this learned distribution.

---

!!! note "Quiz"
**9. The VAE is trained by maximizing the Evidence Lower Bound (ELBO). The ELBO objective represents a trade-off between which two quantities?**
*   A. Training speed and model size.
*   B. The number of layers and the number of neurons.
*   C. A reconstruction term (energy) and a KL-divergence regularization term (entropy).
*   D. The learning rate and the batch size.

??? info "See Answer"
    **C. A reconstruction term (energy) and a KL-divergence regularization term (entropy).**
    
    **Explanation:** The VAE must balance the fidelity of its reconstructions (low energy) with the simplicity and regularity of its latent space (low entropy cost via the KL term), which is a direct manifestation of the free-energy principle.

---

!!! note "Quiz"
**10. What is the "reparameterization trick" essential for training VAEs?**
*   A. A method for re-initializing the weights of the network periodically.
*   B. A way to move the stochastic sampling operation out of the computational graph, allowing gradients to flow through the encoder's mean and variance outputs.
*   C. A technique for reducing the number of parameters in the decoder.
*   D. A trick to make the KL-divergence term always equal to zero.

??? info "See Answer"
    **B. A way to move the stochastic sampling operation out of the computational graph, allowing gradients to flow through the encoder's mean and variance outputs.**
    
    **Explanation:** By expressing the latent sample as $\mathbf{z} = \mathbf{\mu} + \mathbf{\sigma} \odot \mathbf{\epsilon}$, the random part ($\epsilon$) is external, making the path from the parameters ($\mu, \sigma$) to the loss differentiable.

---

!!! note "Quiz"
**11. How do Generative Adversarial Networks (GANs) bypass the problem of the intractable partition function?**
*   A. They use a VAE to approximate the partition function.
*   B. They frame learning as a two-player minimax game between a Generator and a Discriminator, avoiding the need to explicitly calculate the probability density.
*   C. They are restricted to datasets where the partition function is 1.
*   D. They use Contrastive Divergence to estimate the partition function.

??? info "See Answer"
    **B. They frame learning as a two-player minimax game between a Generator and a Discriminator, avoiding the need to explicitly calculate the probability density.**
    
    **Explanation:** The discriminator provides a learned loss function, guiding the generator without needing to evaluate the likelihood, thus bypassing the $Z_\theta$ problem entirely.

---

!!! note "Quiz"
**12. In the context of GANs, what is the Nash Equilibrium?**
*   A. The point where the Generator's loss is zero and the Discriminator's loss is infinite.
*   B. The point where the Generator produces samples that are indistinguishable from real data, and the Discriminator's accuracy is no better than random guessing (0.5).
*   C. The point where both the Generator and Discriminator networks have the same number of parameters.
*   D. The first step of the training process.

??? info "See Answer"
    **B. The point where the Generator produces samples that are indistinguishable from real data, and the Discriminator's accuracy is no better than random guessing (0.5).**
    
    **Explanation:** At this equilibrium, the generator's distribution $p_g$ has converged to the data distribution $p_{data}$, and neither player can unilaterally improve their outcome.

---

!!! note "Quiz"
**13. What is the physical analogy for the GAN training process?**
*   A. A system undergoing radioactive decay.
*   B. Two competing fields seeking a thermodynamic balance or equilibrium.
*   C. A particle moving in a fixed gravitational potential.
*   D. The process of crystallization from a liquid.

??? info "See Answer"
    **B. Two competing fields seeking a thermodynamic balance or equilibrium.**
    
    **Explanation:** The generator acts as a source creating new configurations, while the discriminator acts as a critic imposing constraints. The equilibrium is where the generative pressure is balanced by the critical pressure.

---

!!! note "Quiz"
**14. What is the core idea behind Diffusion Models?**
*   A. They learn to add noise to an image until it becomes a perfect Gaussian distribution.
*   B. They learn to reverse a fixed, gradual noising process (diffusion), effectively learning to denoise a random signal back into a structured sample.
*   C. They simulate the diffusion of particles in a fluid to generate images.
*   D. They are a type of GAN with three competing networks.

??? info "See Answer"
    **B. They learn to reverse a fixed, gradual noising process (diffusion), effectively learning to denoise a random signal back into a structured sample.**
    
    **Explanation:** This process is analogous to time-reversed thermodynamics, where the model learns to decrease entropy and restore order from chaos.

---

!!! note "Quiz"
**15. The training objective for a diffusion model is often based on "score matching." What is the network trained to predict?**
*   A. The final, clean image from the noisy input.
*   B. The class label of the noisy image.
*   C. The amount of noise that was added to the data at a particular timestep.
*   D. The value of the partition function.

??? info "See Answer"
    **C. The amount of noise that was added to the data at a particular timestep.**
    
    **Explanation:** By predicting the noise $\epsilon$, the model is implicitly learning the score function ($\nabla_{\mathbf{x}} \log q(\mathbf{x}_t)$), which is the gradient required to reverse the diffusion process.

---

!!! note "Quiz"
**16. What thermodynamic process is analogous to the forward process (adding noise) in a diffusion model?**
*   A. A system approaching absolute zero.
*   B. The increase of entropy as a system moves towards thermal equilibrium (disorder).
*   C. A phase transition from a gas to a solid.
*   D. The conversion of work into heat with 100% efficiency.

??? info "See Answer"
    **B. The increase of entropy as a system moves towards thermal equilibrium (disorder).**
    
    **Explanation:** The gradual addition of noise systematically destroys information and increases the entropy of the data, analogous to the second law of thermodynamics.

---

!!! note "Quiz"
**17. Comparing generative paradigms, which model is most directly analogous to learning a Boltzmann distribution by sculpting an energy landscape?**
*   A. Generative Adversarial Network (GAN)
*   B. Variational Autoencoder (VAE)
*   C. Energy-Based Model (EBM) / Boltzmann Machine
*   D. A simple Feedforward Neural Network

??? info "See Answer"
    **C. Energy-Based Model (EBM) / Boltzmann Machine**
    
    **Explanation:** EBMs are the most direct implementation of the energy-probability duality $p(\mathbf{x}) \propto e^{-E_\theta(\mathbf{x})}$, making them a literal model of a statistical mechanics system.

---

!!! note "Quiz"
**18. Why do GANs often produce sharper images compared to VAEs?**
*   A. VAEs use a more complex loss function.
*   B. GANs use an adversarial discriminator as a learned, dynamic loss function that is better at penalizing perceptual inconsistencies, while VAEs often optimize a pixel-wise error (like MSE) which encourages blurry averages.
*   C. GANs are trained for more epochs.
*   D. VAEs can only be trained on low-resolution images.

??? info "See Answer"
    **B. GANs use an adversarial discriminator as a learned, dynamic loss function that is better at penalizing perceptual inconsistencies, while VAEs often optimize a pixel-wise error (like MSE) which encourages blurry averages.**
    
    **Explanation:** The adversarial loss is more aligned with perceptual reality, rewarding the generator for producing high-frequency details that can fool the discriminator.

---

!!! note "Quiz"
**19. The relationship between an Autoencoder (AE) and a Restricted Boltzmann Machine (RBM) can be described as:**
*   A. They are identical architectures with different names.
*   B. An AE is a probabilistic version of an RBM.
*   C. An AE's deterministic behavior is analogous to an RBM in the zero-temperature limit ($T \to 0$).
*   D. An RBM is a type of AE with more layers.

??? info "See Answer"
    **C. An AE's deterministic behavior is analogous to an RBM in the zero-temperature limit ($T \to 0$).**
    
    **Explanation:** As thermal noise is removed, the stochastic binary units of the RBM collapse to their most probable, deterministic state, mimicking the feedforward mapping of an AE.

---

!!! note "Quiz"
**20. In the context of a $\beta$-VAE, what is the effect of setting the hyperparameter $\beta > 1$?**
*   A. It decreases the importance of the reconstruction loss, allowing the model to learn a more disentangled latent space.
*   B. It increases the learning rate for the encoder.
*   C. It forces the model to prioritize perfect reconstruction at the expense of latent space structure.
*   D. It removes the KL-divergence term from the loss function entirely.

??? info "See Answer"
    **A. It decreases the importance of the reconstruction loss, allowing the model to learn a more disentangled latent space.**
    
    **Explanation:** A larger $\beta$ puts more weight on the KL-divergence term, penalizing complex latent distributions more heavily and encouraging the encoder to find simpler, more regular representations, even if it means sacrificing some reconstruction fidelity.

---

!!! note "Quiz"
**21. What is the conceptual link between generative modeling and the variational principle in quantum mechanics?**
*   A. Both are used exclusively for image generation.
*   B. Both rely on minimizing an energy-related functional (Variational Free Energy in EBMs, expected Hamiltonian energy in QM) to find an optimal configuration.
*   C. Both require the use of a Generative Adversarial Network.
*   D. There is no known link between the two fields.

??? info "See Answer"
    **B. Both rely on minimizing an energy-related functional (Variational Free Energy in EBMs, expected Hamiltonian energy in QM) to find an optimal configuration.**
    
    **Explanation:** This shared variational core allows neural networks (like RBMs) to be used as a flexible ansatz for quantum wavefunctions, a concept known as Neural Quantum States (NQS).

---

!!! note "Quiz"
**22. If you use a trained EBM to perform MCMC sampling (e.g., Metropolis-Hastings), what are you effectively doing?**
*   A. Further training the model's weights.
*   B. Generating new samples by exploring the learned low-energy basins of the energy landscape.
*   C. Calculating the exact value of the partition function.
*   D. Classifying the input data.

??? info "See Answer"
    **B. Generating new samples by exploring the learned low-energy basins of the energy landscape.**
    
    **Explanation:** MCMC is a method for "thermal exploration" of the state space, which naturally gravitates towards the low-energy (high-probability) regions defined by the learned energy function.

---

!!! note "Quiz"
**23. Which generative model explicitly defines a forward process of adding noise and a learned reverse process of removing it?**
*   A. Boltzmann Machine
*   B. Variational Autoencoder (VAE)
*   C. Generative Adversarial Network (GAN)
*   D. Diffusion Model

??? info "See Answer"
    **D. Diffusion Model**
    
    **Explanation:** This two-part structure of a fixed diffusion process and a learned denoising process is the defining characteristic of diffusion models.

---

!!! note "Quiz"
**24. In the code demo for the toy EBM, the loss function is `loss = Eb - Em`. What do `Eb` and `Em` represent?**
*   A. `Eb` is the energy of the encoder, `Em` is the energy of the decoder.
*   B. `Eb` is the average energy of the real data (positive phase), `Em` is the average energy of fake/model samples (negative phase).
*   C. `Eb` is the energy before the update, `Em` is the energy after the update.
*   D. `Eb` is the energy of the biases, `Em` is the energy of the main weights.

??? info "See Answer"
    **B. `Eb` is the average energy of the real data (positive phase), `Em` is the average energy of fake/model samples (negative phase).**
    
    **Explanation:** Minimizing this difference pushes the energy of real data down (`Eb`) and the energy of fake data up (`Em`), sculpting the energy landscape correctly.

---

!!! note "Quiz"
**25. How does the concept of "Neural Quantum States" (NQS) utilize generative models?**
*   A. It uses GANs to generate images of quantum phenomena.
*   B. It uses a neural network, such as an RBM, as a variational ansatz to represent the complex wavefunction of a many-body quantum system.
*   C. It uses VAEs to compress quantum data.
*   D. It uses diffusion models to simulate quantum tunneling.

??? info "See Answer"
    **B. It uses a neural network, such as an RBM, as a variational ansatz to represent the complex wavefunction of a many-body quantum system.**
    
    **Explanation:** The network's parameters are optimized to minimize the expectation value of the physical Hamiltonian, allowing the model to find the system's ground state wavefunction. This directly applies the principles of generative modeling to solve problems in quantum physics.

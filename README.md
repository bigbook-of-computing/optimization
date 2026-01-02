# Data, Optimization & Machine Learning

[![Documentation](https://img.shields.io/badge/docs-live-brightgreen)](https://bigbookofcomputing.github.io)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![MkDocs](https://img.shields.io/badge/Built%20with-MkDocs-blue)](https://www.mkdocs.org/)

> **Volume III** of the *Big Book of Computing* series

## 📖 About

**Data, Optimization & Machine Learning** bridges the gap between computational simulation and intelligent insight. This volume tackles the "so what?" problem—after generating terabytes of simulation data, how do we extract meaning, discover patterns, and build predictive models?

This book presents a unified framework connecting statistical learning, optimization theory, and modern AI through the lens of physics and applied mathematics. From dimensionality reduction to deep learning, from gradient descent to neural quantum states, we explore how optimization and learning are fundamentally about navigating energy landscapes and discovering structure in high-dimensional spaces.

## 🎯 Core Philosophy

This volume builds on a key insight: **optimization and learning are two sides of the same coin**. Whether minimizing energy in physical systems, loss functions in neural networks, or uncertainty in probabilistic models, we're always searching for "good states" in complex landscapes.

The book emphasizes:
- **Physical intuition** — Understanding learning through energy, dynamics, and geometry
- **Unified perspective** — Connecting statistical mechanics, optimization, and AI
- **Practical implementation** — From theory to working code
- **Modern frontiers** — Physics-informed ML, neural quantum states, and transformers

## 🎯 What's Inside

### Part I: The Geometry of Data (Chapters 1-3)

Understanding the shape and structure of high-dimensional data.

- **Chapter 1**: From Simulation to Data — Trajectories to datasets, manifolds, and metrics
- **Chapter 2**: Statistics & Probability in High Dimensions — Distributions, entropy, curse of dimensionality
- **Chapter 3**: Dimensionality Reduction & Clustering — PCA, t-SNE, UMAP, k-means, hierarchical clustering

### Part II: Optimization as Physics (Chapters 4-8)

Every learning process is optimization on an energy landscape.

- **Chapter 4**: The Optimization Landscape — Convex vs. non-convex, local vs. global minima
- **Chapter 5**: Gradient Methods — Gradient descent and SGD as physical relaxation
- **Chapter 6**: Advanced Gradient Dynamics — Momentum, RMSProp, Adam as damped motion
- **Chapter 7**: Stochastic & Heuristic Optimization — Simulated annealing, genetic algorithms, swarm methods
- **Chapter 8**: Combinatorial Optimization and QUBO — Discrete optimization and the Ising-QUBO connection

### Part III: Learning as Inference (Chapters 9-11)

From optimization to reasoning—learning as belief update.

- **Chapter 9**: Bayesian Thinking and Inference — Posterior inference and MAP estimation
- **Chapter 10**: Regression & Classification — Linear models, regularization, bias-variance tradeoff
- **Chapter 11**: Graphical Models & Probabilistic Graphs — Bayes nets, MRFs, and belief propagation

### Part IV: Deep Learning as Representation (Chapters 12-15)

Hierarchical neural architectures as energy minimizers and function approximators.

- **Chapter 12**: The Perceptron and Neural Foundations — From linear classifiers to multilayer networks
- **Chapter 13**: Hierarchical Representation Learning — CNNs, RNNs, and autoencoders
- **Chapter 14**: Energy-Based and Generative Models — Boltzmann machines, VAEs, and GANs
- **Chapter 15**: Reinforcement Learning and Control — Agents, rewards, Q-learning, and policy gradients

### Part V: The Frontier—Physics ↔ AI (Chapters 16-19)

Where learning, optimization, and physical law converge.

- **Chapter 16**: Physics-Informed Neural Networks (PINNs) — Neural nets satisfying PDEs by construction
- **Chapter 17**: Neural Quantum States (NQS) — ANN ansatz for wavefunctions and variational Monte Carlo
- **Chapter 18**: Graph Neural Networks (GNNs) — Message-passing, permutation invariance, molecular AI
- **Chapter 19**: Transformers and Global Correlation — Self-attention as learned long-range coupling

## 🚀 Getting Started

### View the Book Online

The complete book is available online at: **[https://bigbookofcomputing.github.io](https://bigbookofcomputing.github.io)**

### Build Locally

To build and serve the documentation locally:

1. **Clone the repository**
   ```bash
   git clone https://github.com/bigbookofcomputing/optimization.git
   cd optimization
   ```

2. **Install dependencies**
   ```bash
   pip install mkdocs-material
   pip install mkdocs-minify-plugin
   ```

3. **Serve locally**
   ```bash
   mkdocs serve
   ```
   
   Then open your browser to `http://127.0.0.1:8000`

4. **Build static site**
   ```bash
   mkdocs build
   ```

5. **Deploy to GitHub Pages**
   ```bash
   mkdocs gh-deploy
   ```

## 📚 Enhanced Learning Structure

Each chapter provides multiple learning pathways:

- **📖 Essay** — Deep theoretical understanding with physical intuition
- **📘 WorkBook** — Hands-on exercises to build mastery
- **💻 CodeBook** — Runnable implementations with detailed comments
- **📝 Quizzes** — Test your understanding of key concepts
- **💼 Interviews** — Practice problems for technical interviews
- **🚀 Projects** — End-to-end applications integrating multiple concepts
- **🔬 Research** — Connections to cutting-edge research and open problems

This multi-faceted approach ensures deep understanding from theory to practice.

## 🔗 Key Connections & Unifying Themes

### Physics ↔ Machine Learning Analogies

| Physics Concept | ML Equivalent | Connection |
|----------------|---------------|------------|
| Energy minimization | Loss minimization | Same mathematical framework |
| Statistical mechanics | Probabilistic modeling | Partition functions ↔ normalizing constants |
| Phase transitions | Sudden learning dynamics | Critical points in loss landscape |
| Spin systems | Neural networks | Ising model ↔ Hopfield network |
| Hamiltonian dynamics | Gradient flow | Energy conservation ↔ learning dynamics |

### Cross-Volume Integration

- **Volume I foundations** — Numerical methods for gradients, ODEs, and PDEs
- **Volume II simulations** — Monte Carlo, molecular dynamics, and stochastic processes
- **Volume III learning** — Extracting patterns and optimizing from simulated data
- **Volume IV quantum** (coming) — Quantum annealing, variational algorithms, and quantum ML

## 🛠️ Technologies & Prerequisites

### Technologies Used
- **Python** — Primary implementation language
- **NumPy/SciPy** — Numerical computing and optimization
- **PyTorch/TensorFlow** — Deep learning frameworks
- **Scikit-learn** — Classical machine learning
- **MkDocs Material** — Documentation framework
- **MathJax** — Mathematical typesetting

### Prerequisites

**Essential:**
- Linear algebra (vectors, matrices, eigenvalues)
- Multivariable calculus (gradients, chain rule)
- Basic probability and statistics
- Python programming

**Helpful:**
- Volume I (Foundation) for numerical methods background
- Volume II (Simulation) for stochastic processes context
- Basic familiarity with physics concepts

## 🎓 Who Should Read This Book?

This book is designed for:

- **Physics students** transitioning to machine learning and data science
- **ML practitioners** seeking deeper physical intuition for algorithms
- **Data scientists** working with scientific or simulation data
- **Researchers** at the intersection of physics and AI
- **Engineers** applying optimization in complex systems
- **Anyone** curious about the deep connections between learning and physics

## 💡 What Makes This Book Unique?

1. **Physics-first perspective** — ML explained through energy landscapes and statistical mechanics
2. **Unified framework** — Optimization and learning as a single conceptual journey
3. **Modern frontiers** — PINNs, NQS, GNNs, and transformers from first principles
4. **Simulation to insight** — Directly addresses the data-driven discovery pipeline
5. **Multiple learning modalities** — Essays, workbooks, code, quizzes, interviews, projects, and research connections

## 🤝 Contributing

We welcome contributions! Whether it's:

- Improving explanations or fixing errors
- Adding new examples or visualizations
- Suggesting additional topics or applications
- Contributing code implementations
- Reporting issues

Please feel free to open an issue or submit a pull request.

## 📄 License

This work is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🌟 About the Big Book of Computing

This is **Volume III** of the *Big Book of Computing* series:

- **Volume I**: [Foundation of Computational Science](https://github.com/bigbookofcomputing/foundation) — Numerical methods and computational foundations
- **Volume II**: [Simulating Complex Systems](https://github.com/bigbookofcomputing/simulation) — Monte Carlo, dynamics, and agent-based models
- **Volume III**: **Data, Optimization & Machine Learning** — From data to intelligence (this volume)
- **Volume IV**: Quantum Computing *(coming soon)* — Quantum algorithms and quantum machine learning

## 📧 Contact

- **Website**: [https://bigbookofcomputing.github.io](https://bigbookofcomputing.github.io)
- **GitHub**: [https://github.com/bigbookofcomputing](https://github.com/bigbookofcomputing)
- **Twitter**: [@bigbookofcomputing](https://x.com/bigbookofcomputing)

---

**Built with ❤️ for those who see learning as energy minimization and intelligence as emergent order**

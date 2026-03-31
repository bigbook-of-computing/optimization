# **Introduction**

Optimization is the study of how to choose well under constraints. In computational work, that simple idea appears everywhere: minimizing energy, fitting models, tuning parameters, estimating hidden variables, planning actions, and balancing tradeoffs between accuracy, cost, stability, and time.

This matters because many of the most important scientific and engineering problems are not solved by closed-form formulas. Instead, they are posed as search problems over large spaces. We define an objective, describe the feasible set, and then use algorithms to locate points that are optimal or at least useful. In that sense, optimization is not a specialized afterthought. It is one of the central operating principles of modern computation.

## Why Optimization Matters

Optimization provides a common language across disciplines.

- In physics, stable states often correspond to minima of an energy functional.
- In statistics, estimation is frequently written as likelihood maximization or loss minimization.
- In machine learning, training is the repeated optimization of parameters over large datasets.
- In engineering, design choices are constrained by resources, safety margins, and performance objectives.
- In operations research, planning and allocation problems become discrete optimization tasks.

The same core ideas therefore reappear under many names: objective, cost, loss, energy, utility, posterior, value function, or risk.

## The Central Difficulty

Optimization is conceptually simple and computationally difficult. Real objectives are often high-dimensional, noisy, constrained, non-convex, and expensive to evaluate. Some problems admit elegant convex theory and reliable convergence guarantees. Others involve rugged landscapes, combinatorial explosions, or stochastic updates that make exact solutions unrealistic.

Three recurring tensions shape the field:

- Local structure versus global behavior.
- Computational cost versus solution quality.
- Mathematical guarantees versus practical performance.

This book treats those tensions directly rather than hiding them behind idealized examples.

## The Main Viewpoints In This Book

The volume approaches optimization from several complementary directions.

First, it begins with data geometry and probability, because objectives are shaped by the structure of the data and models that generate them.

Second, it develops optimization as a dynamical process. Gradient descent, momentum methods, annealing, and heuristic search can all be read as rules for moving across a landscape under different assumptions about smoothness, noise, and available information.

Third, it shows how inference and learning are built on top of optimization. Regression, classification, graphical models, and neural networks all rely on solving parameter selection problems, sometimes explicitly and sometimes implicitly.

Finally, it moves into frontier topics where optimization interacts with scientific structure, including physics-informed learning, neural quantum states, graph-based models, and transformer architectures.

## What You Should Expect As A Reader

You do not need to begin as a specialist, but you should expect the subject to become mathematical. Optimization requires fluency with functions, vectors, matrices, derivatives, probability, and algorithms. Some chapters also rely on linear algebra, numerical computation, and modeling intuition.

This book therefore emphasizes:

- Clear definitions before algorithmic recipes.
- Geometric and dynamical intuition alongside formal statements.
- Repeated links between mathematical objectives, computational procedures, and applications.
- A realistic view of where optimization works cleanly and where it becomes genuinely hard.

## How To Read The Volume

If you are new to the subject, read the first eight chapters in sequence. They establish the language of data geometry, probability, landscapes, gradients, and combinatorial structure that later chapters assume.

If your main interest is machine learning, you can treat Chapters 9 to 15 as the bridge from optimization foundations to modeling practice. If your interest is current research directions, Chapters 16 to 19 show how optimization continues to shape modern scientific and AI systems.

Before moving on, it is useful to skim the [Contents](contents.md) page for the full map of the volume.
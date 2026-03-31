# Source: Optimization/chapter-10/essay.md -- Block 2

# Synthetic classification data (2D input, binary output)
np.random.seed(0)
X = np.random.randn(200,2)
# Create a hidden linear boundary to generate labels: y = 1 if 0.5*x1 - 0.7*x2 > 0
y = (X[:,0]*0.5 + X[:,1]*(-0.7) > 0).astype(int)
# Fit the logistic classifier
clf = LogisticRegression().fit(X, y)

plt.figure(figsize=(9, 6))
plt.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm', alpha=0.6, label='Data by Class')

# Define a grid for plotting the decision boundary
xx, yy = np.meshgrid(np.linspace(-3,3,100), np.linspace(-3,3,100))
input_grid = np.c_[xx.ravel(), yy.ravel()]
# Predict probability of class 1 across the grid
Z = clf.predict_proba(input_grid)[:,1].reshape(xx.shape)

# Plot the decision boundary (where p=0.5, or w^T x = 0, Section 10.6)
plt.contour(xx, yy, Z, levels=[0.5], colors='k', linestyles='-', linewidths=3, label='Decision Boundary')

plt.title('Logistic Regression Decision Boundary')
plt.xlabel('Feature $x_1$')
plt.ylabel('Feature $x_2$')
plt.show()

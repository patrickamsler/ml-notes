# Machine Learning

- [Machine Learning](#machine-learning)
  - [Overview](#overview)
    - [Supervised Learning](#supervised-learning)
    - [Unsupervised Learning](#unsupervised-learning)
    - [Other Methods](#other-methods)
  - [Linear Regression](#linear-regression)
    - [Model](#model)
    - [Cost Function (Squared Error):](#cost-function-squared-error)
    - [Gradient Descent](#gradient-descent)
    - [Multiple Linear Regression](#multiple-linear-regression)
      - [Dot Product](#dot-product)
      - [Vectorization](#vectorization)


## Overview

### Supervised Learning
- Learns given the right answers
- labeled data x -> y
- regression, classification

### Unsupervised Learning
- Learns without the right answers
- Find structure in data
- clustering, dimensionality reduction, anomaly detection

### Other Methods
 - Reinforcement Learning
 - Recommender Systems

## Linear Regression

### Model
$w$ and $b$ are the parameters of the model. The model is a linear function of the input $x$.

$$
\hat{y} = f_{wb}(x) = wx + b
$$

- **$w$**: The weight or slope of the line, which shows how much $y$ changes for a unit change in $x$.
- **$b$**: The bias or intercept, representing the value of $y$ when $x = 0$
- **$\hat{y}$**: The predicted output or dependent variable.
- **$x$**: The input or independent variable.

### Cost Function (Squared Error):
The cost function $J(w, b)$ measures the average squared difference between the predicted values $\hat{y}$ and the actual values $y$. It is defined as:

$$
J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{wb}(x^{(i)}) - y^{(i)} \right)^2
$$

$$
J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)^2
$$

- $m$ is the number of training examples.
- $f_{wb}(x^{(i)})$ is the predicted value of the model for the $i$-th training example.
- $y^{(i)}$ is the actual value for the $i$-th training example.

### Gradient Descent

The goal of training a model is to find the parameters $w$ and $b$ that minimize the cost function $J(w, b)$.

At each iteration, the parameters $w$ and $b$ are updated using the following rules:

$$
w := w - \alpha \frac{\partial}{\partial w} J(w, b)
$$

$$
b := b - \alpha \frac{\partial}{\partial b} J(w, b)
$$

- $\alpha$ is the learning rate, which controls the size of the steps taken towards the minimum.
- $\frac{\partial}{\partial w} J(w, b)$ is the partial derivative of the cost function with respect to $w$
- $\frac{\partial}{\partial b} J(w, b)$ is the partial derivative of the cost function with respect to $b$

Partial Derivative with Respect to $w$:

$$
\frac{\partial}{\partial w} J(w, b) = \frac{1}{m} \sum_{i=1}^{m} \left( wx^{(i)} + b - y^{(i)} \right) x^{(i)}
$$

Partial Derivative with Respect to $b$:

$$
\frac{\partial}{\partial b} J(w, b) = \frac{1}{m} \sum_{i=1}^{m} \left( wx^{(i)} + b - y^{(i)} \right)
$$

Learning Rate
- The learning rate $\alpha$ is a parameter that controls the size of the steps taken during gradient descent. 
- If the learning rate is too small, it will take longer to converge.
- If the learning rate is too large, it may overshoot the minimum and fail to converge.
- The squared error function is convex (only one minimum), so gradient descent will always converge to the global minimum.
- Other cost functions may have multiple local minima. It is not guaranteed that gradient descent will converge to the global minimum. So multiple starting points are used to find the best minimum.

Batch Gradient Descent:
- In batch gradient descent, the parameters are updated after computing the gradient of the cost function for the entire training set.
- Other methods use a subset of the training set to compute the gradient. These methods are called stochastic gradient descent and mini-batch gradient descent.

### Multiple Linear Regression

The model predicts the output based on multiple input features $x_1, x_2, …, x_n$. The parameters $w_1, w_2, …, w_n$ represent the weights for each feature, and $b$ is the bias term.

$$
\hat{y} = f_{wb}(\vec{x}) = \vec{w} \cdot \vec{x}  = w_1x_1 + w_2x_2 + \dots + w_nx_n + b
$$

- n: the number of input features.
- $w_1, w_2, …, w_n$: The weights corresponding to each input feature, showing the contribution of each feature to the prediction.
- $b$: The bias or intercept term.
- $\hat{y}$: The predicted output based on all input features (independent variables) $x_1, x_2, …, x_n$.

#### Dot Product
Weights and features are calculated using the dot product. For two vectors with the same dimension **$\vec{w}$** and **$\vec{x}$**, each with $n$ entries, the dot product is calculated as:

$$
\vec{w} \cdot \vec{x} = w_1x_1 + w_2x_2 + \dots + w_nx_n
$$

Let **$\vec{w}$** and **$\vec{x}$** be two vectors with 4 entries:

$$
\vec{w} = [w_1, w_2, w_3, w_4] = [2, -1, 0.5, 3]
$$

$$
\vec{x} = [x_1, x_2, x_3, x_4] = [1, 0.5, -2, 4]
$$

$$
\vec{w} \cdot \vec{x} = (2)(1) + (-1)(0.5) + (0.5)(-2) + (3)(4) = 12.5
$$

#### Vectorization

Vectorization allows for efficient computation by applying operations to entire arrays. The dot product can be build by CPU and GPU that run in parallel.

```python
# Define the vectors
w = np.array([2, -1, 0.5, 3])
x = np.array([1, 0.5, -2, 4])

# Compute the dot product
dot_product = np.dot(w, x)
```
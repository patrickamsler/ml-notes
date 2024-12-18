# Machine Learning

- [Machine Learning](#machine-learning)
  - [Linear Regression](#linear-regression)
    - [Model](#model)
    - [Cost Function (Squared Error):](#cost-function-squared-error)
    - [Gradient Descent](#gradient-descent)
    - [Multiple Linear Regression](#multiple-linear-regression)
      - [Vectorization](#vectorization)
      - [Cost Function for Multiple Linear Regression](#cost-function-for-multiple-linear-regression)
      - [Gradient Descent for Multiple Linear Regression](#gradient-descent-for-multiple-linear-regression)
    - [Normal Equation](#normal-equation)
    - [Feature Scaling](#feature-scaling)
      - [Mean Normalization](#mean-normalization)
      - [Z-Score Normalization](#z-score-normalization)
    - [Feature Engineering](#feature-engineering)
      - [Feature Selection](#feature-selection)
    - [Polynomial Regression](#polynomial-regression)
  - [Classification with Logistic Regression](#classification-with-logistic-regression)
    - [Sigmoid Function](#sigmoid-function)
    - [Logistic Regression Model](#logistic-regression-model)
    - [Decision Boundary](#decision-boundary)
    - [Cost Function for Logistic Regression](#cost-function-for-logistic-regression)
    - [Gradient Descent for Logistic Regression](#gradient-descent-for-logistic-regression)
  - [Neural Networks](#neural-networks)
    - [Neural Network with Tensorflow](#neural-network-with-tensorflow)
    - [Activation Functions](#activation-functions)
    - [Multiclass Classification](#multiclass-classification)
      - [Softmax Function](#softmax-function)
      - [Cost Function for Multiclass Classification](#cost-function-for-multiclass-classification)
      - [Multiclass Classification with Tensorflow](#multiclass-classification-with-tensorflow)
    - [Multilable Classification](#multilable-classification)
    - [Adam Algorithm](#adam-algorithm)
    - [Layers in a Neural Network](#layers-in-a-neural-network)
      - [Dense Layer](#dense-layer)
      - [Convolutional Layer](#convolutional-layer)
  - [Evaluating the Model](#evaluating-the-model)
    - [The Problem of Overfitting and Underfitting](#the-problem-of-overfitting-and-underfitting)
    - [Regularization to Reduce Overfitting](#regularization-to-reduce-overfitting)
      - [Neural network regularization](#neural-network-regularization)
    - [Train test prodcedure for linear regression](#train-test-prodcedure-for-linear-regression)
    - [Train test prodcedure for logistic regression](#train-test-prodcedure-for-logistic-regression)
    - [Cross-Validation](#cross-validation)
      - [K-Fold Cross-Validation](#k-fold-cross-validation)
      - [R2 Score](#r2-score)
    - [Diagnosing Bias and Variance](#diagnosing-bias-and-variance)
    - [Baseline level of performance](#baseline-level-of-performance)
    - [Learning Curves](#learning-curves)
    - [Bias and Variance in Neural Networks](#bias-and-variance-in-neural-networks)
    - [Precision and Recall](#precision-and-recall)
      - [Trade off between precision and recall](#trade-off-between-precision-and-recall)
      - [F1 Score](#f1-score)
  - [Machine Learining Development Process](#machine-learining-development-process)
    - [Data Augmentation](#data-augmentation)
    - [Data Synthesis](#data-synthesis)
    - [Transfer Learning](#transfer-learning)
  - [Decision Trees](#decision-trees)
    - [Entropy as a measure of impurity](#entropy-as-a-measure-of-impurity)
    - [Information Gain](#information-gain)
    - [Decision Tree Learning Algorithm (Recursive Splitting)](#decision-tree-learning-algorithm-recursive-splitting)
    - [Features with Multiple Classes](#features-with-multiple-classes)
      - [One-Hot Encoding](#one-hot-encoding)
    - [Splitting Continuous Variables](#splitting-continuous-variables)
    - [Regression with Decision Trees](#regression-with-decision-trees)
    - [Tree Ensembles](#tree-ensembles)
      - [Bagging (Bootstrap Aggregating)](#bagging-bootstrap-aggregating)
      - [Random Forest Algorithm](#random-forest-algorithm)
      - [Gradient Boosting](#gradient-boosting)
      - [XGBoost](#xgboost)
      - [Decision Trees vs Neural Networks](#decision-trees-vs-neural-networks)
  - [Clustering](#clustering)
    - [K-Means Clustering](#k-means-clustering)
  - [Anomaly Detection](#anomaly-detection)
    - [Gaussian Distribution](#gaussian-distribution)
    - [Density Estimation](#density-estimation)
    - [Anomaly Detection Evaluation](#anomaly-detection-evaluation)
    - [Anomaly Detection vs Supervised Learning](#anomaly-detection-vs-supervised-learning)
    - [Feature Selection in Anomaly Detection](#feature-selection-in-anomaly-detection)
  - [Recommender Systems](#recommender-systems)
    - [Collaborative Filtering](#collaborative-filtering)
      - [Collaborative Filtering Algorithm](#collaborative-filtering-algorithm)
      - [Binary Labels](#binary-labels)
      - [Mean Normalization](#mean-normalization-1)
      - [Finding Similar Items](#finding-similar-items)
      - [Collaborative Filtering in Tensorflow](#collaborative-filtering-in-tensorflow)
      - [Limitations of Collaborative Filtering](#limitations-of-collaborative-filtering)
    - [Content-Based Filtering](#content-based-filtering)
      - [Retrival and Ranking](#retrival-and-ranking)
  - [Reinforcement Learning](#reinforcement-learning)
    - [Markov Decision Process](#markov-decision-process)
    - [State Action Value Function](#state-action-value-function)
    - [Deep Reinforcement Learning](#deep-reinforcement-learning)

## Linear Regression

Supervised learning algorithm that learns from labeled data to predict the output for new, unseen data. It models the relationship between the input and output variables as a linear function. 

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
- Squaring the differences ensures all errors are positive, which avoids negative and positive errors canceling each other out.
- It heavily penalizes larger errors, making the model more sensitive to significant deviations.

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

The derivative of the cost function gives the direction of the steepest ascent. As closer to the minimum, the gradient becomes smaller, and the steps taken are smaller. At the minimum, the gradient is zero, and the parameters do not change.

Partial Derivative with Respect to $w$:

$$
\frac{\partial}{\partial w} J(w, b) = \frac{1}{m} \sum_{i=1}^{m} \left( wx^{(i)} + b - y^{(i)} \right) x^{(i)}
$$

Partial Derivative with Respect to $b$:

$$
\frac{\partial}{\partial b} J(w, b) = \frac{1}{m} \sum_{i=1}^{m} \left( wx^{(i)} + b - y^{(i)} \right)
$$

For more details, refer to the [Calculating Partial Derivatives of a Linear Function and the Squared Error Cost Function](./calculus.md#calculating-partial-derivatives-of-a-linear-function-and-the-squared-error-cost-function) section.


Learning Rate
- The learning rate $\alpha$ is a parameter that controls the size of the steps taken during gradient descent. 
- If the learning rate is too small, it will take longer to converge.
- If the learning rate is too large, it may overshoot the minimum and fail to converge.
- The squared error function is convex (only one minimum), so gradient descent will always converge to the global minimum.
- Other cost functions may have multiple local minima. It is not guaranteed that gradient descent will converge to the global minimum. So multiple starting points are used to find the best minimum.

Batch Gradient Descent:
- In batch gradient descent, the parameters are updated after computing the gradient of the cost function for the entire training set.
- Other methods use a subset of the training set to compute the gradient. These methods are called stochastic gradient descent and mini-batch gradient descent.
  
A learning curve can be used to check if gradient descent is working correctly. The cost function should decrease with each iteration and never increase. 

![alt text](images/cost_vs_itr_2.png)

If it increases, the learning rate may be too large.

![alt text](images/cost_vs_itr_1.png)

### Multiple Linear Regression

The model predicts the output based on multiple input features $x_1, x_2, …, x_n$. The parameters $w_1, w_2, …, w_n$ represent the weights for each feature, and $b$ is the bias term.

Model:

$$
\hat{y} = f_{wb}(\vec{x}) = \vec{w} \cdot \vec{x} + b = w_1x_1 + w_2x_2 + \dots + w_nx_n + b
$$

- n: the number of input features.
- $w_1, w_2, …, w_n$: The weights corresponding to each input feature, showing the contribution of each feature to the prediction.
- $b$: The bias or intercept term.
- $\hat{y}$: The predicted output based on all input features (independent variables) $x_1, x_2, …, x_n$.

Weights and features are calculated using the [Dot Product](linear-algebra.md#dot-product)

Housing price prediction with multiple features:

![alt text](images/housingprice_multiple_features.png)

#### Vectorization

Vectorization allows for efficient computation by applying operations to entire arrays. The dot product can be build by CPU and GPU that run in parallel.

```python
# Define the vectors
w = np.array([2, -1, 0.5, 3])
x = np.array([1, 0.5, -2, 4])

# Compute the dot product
dot_product = np.dot(w, x)
```

#### Cost Function for Multiple Linear Regression

The cost function for multiple linear regression is the same as for simple linear regression, but the model predicts the output based on multiple input features.

$$
J(\vec{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{wb}(\vec{x}^{(i)}) - y^{(i)} \right)^2
$$

Vector Notation:

$$
J(\vec{w}, b) = \frac{1}{2m} \left( \vec{X} \vec{w} + b - \vec{y} \right)^T \left( \vec{X} \vec{w} + b - \vec{y} \right)
$$

- $\vec{X}$ is the matrix of input features.
- $\vec{w}$ is the vector of weights.
- $\vec{y}$ is the vector of actual outputs.

#### Gradient Descent for Multiple Linear Regression


$$
\vec{w} := \vec{w} - \alpha \frac{\partial}{\partial \vec{w}} J(\vec{w}, b)
$$

$$
b := b - \alpha \frac{\partial}{\partial b} J(\vec{w}, b)
$$

Partial Derivative with Respect to $\vec{w}$:

$$
\frac{\partial}{\partial w_1} J(\vec{w}, b) = \frac{1}{m} \sum_{i=1}^{m} \left( f_{wb}(\vec{x}^{(i)}) - y^{(i)} \right) x_1^{(i)}
$$

$$
\frac{\partial}{\partial w_2} J(\vec{w}, b) = \frac{1}{m} \sum_{i=1}^{m} \left( f_{wb}(\vec{x}^{(i)}) - y^{(i)} \right) x_2^{(i)}
$$

...

$$
\frac{\partial}{\partial w_n} J(\vec{w}, b) = \frac{1}{m} \sum_{i=1}^{m} \left( f_{wb}(\vec{x}^{(i)}) - y^{(i)} \right) x_n^{(i)}
$$

Can be written in vector notation:

$$
\frac{\partial}{\partial \vec{w}} J(\vec{w}, b) = \frac{1}{m} \vec{X}^T (\vec{X} \vec{w} + b - \vec{y})
$$

Partial Derivative with Respect to b:

$$
\frac{\partial}{\partial b} J(\vec{w}, b) = \frac{1}{m} \sum_{i=1}^{m} \left( f_{wb}(\vec{x}^{(i)}) - y^{(i)} \right)
$$

### Normal Equation

- The normal equation is an analytical solution to linear regression that minimizes the cost function $J(w, b)$, without the need for iterative optimization algorithms like gradient descent.
- Only works for linear regression and not for other models.
- Solve for w and b by setting the partial derivatives of the cost function to zero.
- May be used by some libraries
- Computationally expensive for large datasets

Formula:

$$
\vec{w} = (\vec{X}^T \vec{X})^{-1} \vec{X}^T \vec{y}
$$

$$
b = \frac{1}{m} \sum_{i=1}^{m} \left( y^{(i)} - \vec{w} \cdot \vec{x}^{(i)} \right)
$$

- $\vec{X}$ is the matrix of input features.
- $\vec{w}$ is the vector of weights.
- $\vec{y}$ is the vector of actual outputs.
- $m$ is the number of training examples.

### Feature Scaling

Feature scaling is a preprocessing step that standardizes the range of independent variables. It is important for algorithms that use gradient descent, where the step size needs to be of similar size for all features in order to converge faster.

Aim for a scale where all the features are in range like -1 to 1 or -0.5 to 0.5.

#### Mean Normalization

$$
x_i = \frac{x_i - \mu}{\text{max}(x) - \text{min}(x)}
$$

Example:

Feature $x_1$ No. Bedrooms (0-5) with mean 2.3 and Area $x_2$ (300-2000) with mean 600 have different ranges.

The normalized value of Bedrooms will be in the range:

$$
\text{Bedrooms} = \frac{\text{Bedrooms} - 2.3}{5-0}
$$

$$
-0.46 \leq x_1 \leq 0.54
$$

The normalized value of Area will be in the range:

$$
\text{Area} = \frac{\text{Area} - 1150}{2000-300}
$$

$$
-0.18 \leq x_2 \leq 0.82
$$

#### Z-Score Normalization

$$
x_i = \frac{x_i - \mu}{\sigma}
$$

- $\mu$ is the mean of the feature.
- $\sigma$ is the standard deviation of the feature.

age vs size of house before and after normalization:

![alt text](images/z_score_narmalization.png)

### Feature Engineering

Feature engineering is the process of creating new features from existing features. It involves combining and transforming features to make them more informative.

$$
f_{wb}(\vec{x}) = w_1x_1 + w_2x_2 + b
$$

where $x_1$ is the length and $x_2$ is the width of a house. 

A new feature $x_3$ can be created by multiplying the length and width to represent the area of the house.

$$
x_3 = x_1 \times x_2
$$

model with new feature:

$$
f_{wb}(\vec{x}) = w_1x_1 + w_2x_2 + w_3x_3 + b
$$

#### Feature Selection

For categorical data, the chi square test can be used to select features. 

During feature selection, each feature is tested with the chi-square statistic. Features with high chi-square scores are selected as they show a significant association with the target variable. 

$$
\chi^2 = \sum \frac{(O - E)^2}{E}
$$

- $O$: Observed frequency, which is the actual frequency of the feature.
- $E$: Expected frequency, the count that would be expecte if there is no association between the feature and the target variable.
- The higher the value of $\chi^2$, the more dependent the feature is on the target variable.

### Polynomial Regression

Polynomial regression is a form of linear regression in which the relationship between the independent variable $x$ and the dependent variable $y$ is modeled as an $n$-th degree polynomial.

$$
f_{wb}(\vec{x}) = w_1x + w_2x^2 + \dots + w_nx^n + b  
$$

It's important to scale the features. The polynomial terms $x^2, x^3$, which are also new features, need to be in the same range as $x$. Otherwise, the features with higher power e.g. $x^3$ will dominate the cost function.

It's also possible to use a model with a root of x or a logarithm of x.

$$
f_{wb}(\vec{x}) = w_0 + w_1x + w_2\sqrt{x}
$$

This function will be steep at the beginning and then flatten out.

## Classification with Logistic Regression

Logistic regression is a supervised learning algorithm used for binary classification. It predicts the probability that an instance belongs to a particular class. The predicted probability is then converted into a binary output.

### Sigmoid Function

![alt text](images/sigmoid_function.png)

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

### Logistic Regression Model

the input $z$ to the sigmoid function is the output of a linear regression model:

$$
z = w_1x_1 + w_2x_2 + \dots + w_nx_n + b
$$


The logistic regression model can be represented as:

$$
f_{wb}(\vec{x}) = \sigma(\vec{w} \cdot \vec{x} + b) = \frac{1}{1 + e^{-(\vec{w} \cdot \vec{x} + b)}}
$$

The model predicts the probability that an instance belongs to the positive class. The predicted probability is then converted into a binary output using a threshold value.

```math
\hat{y} = \begin{cases} 1 & \text{if } P(y=1|\vec{x}) \geq 0.5 \\ 0 & \text{otherwise} \end{cases}
```

For a logistic regression model, $z = \vec{w} \cdot \vec{x} + b$.

- if $\vec{w} \cdot \vec{x} + b \geq 0$, the model predicts $y = 1$
- if $\vec{w} \cdot \vec{x} + b < 0$, the model predicts $y = 0$


### Decision Boundary


Linear decision boundary:

$$
w_1x_1 + w_2x_2 + b = 0
$$

Example:

Given $b = -3$, $w_0 = 1$, and $w_1 = 1$, the model predicts $y = 1$ if $x_1 + x_2 - 3 \geq 0$.


![alt text](images/linear_decision_boundary.png)

Non-linear decision boundary:

$$
w_1x_1^2 + w_2x_2^2 + b = 0
$$

![alt text](images/non_linear_decision_boundary.png)


### Cost Function for Logistic Regression

The squared error cost function used for linear regression is convex. However, when used with the sigmoid function, it becomes non-convex, leading to multiple local minima. For logistic regression, the cost function is the log loss function, which is convex.

- Loss is a measure of the difference of a single example to its target value while the
- Cost is a measure of the losses over the training set.

Loss function:

```math
L(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) = -y^{(i)} \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right)
```

The loss function can be rewritten to be easier to implement.
  
This is a rather formidable-looking equation. It is less daunting when you consider $y^{(i)}$ can have only two values, 0 and 1. One can then consider the equation in two pieces:  

when $y^{(i)} = 0$, the left-hand term is eliminated:

$$
\begin{align}
L(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), 0) &= (-(0) \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - 0\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) \\
&= -\log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right)
\end{align}
$$

and when $y^{(i)} = 1$, the right-hand term is eliminated:

$$
\begin{align}
  L(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), 1) &=  (-(1) \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - 1\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right)\\
  &=  -\log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right)
\end{align}
$$

Logistic loss function for a single example $(\mathbf{x}^{(i)}, y^{(i)})$:

$$
  L(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) = \begin{cases}
    - \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) & \text{if $y^{(i)}=1$}\\
    - \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) & \text{if $y^{(i)}=0$}
  \end{cases}
$$

![alt text](images/logistic_loss_function.png)

Cost function for logistic regression:

$$
\hat{y}^{(i)} = f_{wb}(\vec{x}^{(i)}) = \frac{1}{1 + e^{-(\vec{w} \cdot \vec{x}^{(i)} + b)}}
$$


$$
J(\vec{w}, b) = -\frac{1}{m} \sum_{i=1}^{m} \left( y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right)
$$

### Gradient Descent for Logistic Regression

The parameters $w$ and $b$ are updated simultaneous using the following rules:

$$
\vec{w} := \vec{w} - \alpha \frac{\partial}{\partial \vec{w}} J(\vec{w}, b)
$$

$$
b := b - \alpha \frac{\partial}{\partial b} J(\vec{w}, b)
$$

Partial Derivative with Respect to $\vec{w}$:

$$
\frac{\partial}{\partial w_j} J(\vec{w}, b) = \frac{1}{m} \sum_{i=1}^{m} \left( f_{wb}(\vec{x}^{(i)}) - y^{(i)} \right) x_j^{(i)}
$$

$$
\frac{\partial}{\partial \vec{w}} J(\vec{w}, b) = \frac{1}{m} \vec{X}^T (\vec{X} \vec{w} + b - \vec{y})
$$

Partial Derivative with Respect to $\vec{b}$:

$$
\frac{\partial}{\partial b} J(\vec{w}, b) = \frac{1}{m} \sum_{i=1}^{m} \left( f_{wb}(\vec{x}^{(i)}) - y^{(i)} \right)
$$

The partial derivatives are the same as for linear regression, but the model predicts the output using the sigmoid function.

## Neural Networks

- Inference (forward propagation): Predicting the output for new, unseen data. 
- Training (back propagation): Learning the parameters of the model from labeled data. It is called backpropagation.

![alt text](images/neural_network.png)

Neural network with 3 layers (we do not count the input layer), two hidden layers and the output layer. The first hidden layer has 3 neurons the second hidden layer has 4 neurons.

A neuron has inputs $x_1, x_2, …, x_n$ and weights $w_1, w_2, …, w_n$. The output of the neuron is the weighted sum of the inputs passed through an activation function.

$$
z = w_1x_1 + w_2x_2 + \dots + w_nx_n + b
$$

$$
z = \vec{w} \cdot \vec{x} + b
$$

Activation function (usually sigmoid for binary classification, softmax for multiclass classification, ReLU for hidden layers):

$$
\vec{a} = f(z)
$$

<img src="images/neural_network2.png" alt="Neural Network 2" height="300" />

The output of the activation function is the input to the next layer.


The activation of the \( j \)-th neuron (unit) in the \( l \)-th layer (dense):

$$
a_j^{(l)} = f\left( \vec{w}_{j}^{(l)} \cdot \vec{a}^{(l-1)} + b_j^{(l)} \right)
$$

e.g. second neuron in the first layer (instead of x we use a(0)):

$$
a_2^{(1)} = f\left( \vec{w}_{2}^{(1)} \cdot \vec{a}^{(0)} + b_2^{(1)} \right)
$$

The number of parameters for a layer:

$$
\text{number of inputs} * \text{number of neurons} + \text{number of biases}
$$

For layer 1 in the example with 5 inputs:
- W1 is (5, 4) = 20 weight parameters
- b1 is (4) bias parameters
- total 24 parameters

For layer 3 in the example:
- W3 is (5, 3) = 15 weight parameters
- b3 is (3) bias parameters
- total 18 parameters

### Neural Network with Tensorflow

Neural network with 2 hidden layer and 1 output layer. Epochs (iterations) are the number of times the model sees the training data.

Binary Crossentropy is the loss function for binary classification also known as logistic loss. It is the same as the log loss function used for logistic regression.

Loss function:

$$
L(f_{\vec{w}b}(\vec{x}^{(i)}), y^{(i)}) = (-y^{(i)} \log\left(f_{\vec{w}b}\left( \vec{x}^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - f_{\vec{w}b}\left( \vec{x}^{(i)} \right) \right)
$$

Cost function:

$$
J(\vec{w}, b) = \frac{1}{m} \sum_{i=1}^{m} \left( L(f_{\vec{w}b}(\vec{x}^{(i)}), y^{(i)}) \right)
$$


```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy

# Define the model with 2 hidden layers
model = Sequential([
  Dense(units=25, activation='relu'),
  Dense(units=15, activation='relu'),
  Dense(units=1, activation='sigmoid')
)]

# Compile the model and define the loss function
model.compile(loss=BinaryCrossentropy())

# Train the model with backpropagation
model.fit(X,Y, epochs=100)
```

### Activation Functions

The activation function of a neuron defines when the output of the neuron is activated or not. It introduces non-linearity to the model, allowing it to learn complex patterns in the data. 

If always using linear activation functions, the model would be equivalent to a linear regression model. If all hidden layers have linear activation functions, and the output layer has a sigmoid activation function, the model would be equivalent to logistic regression.

Common activation functions are ReLU for hidden layers and sigmoid for the output layer.

![alt text](images/activation_functions.png)

- **Linear**: Output is the same as the input. Used in the output layer for regression problems. Almost the same as no activation function. It can predict any real number. E.g. predict stock prices where the output can be any positive or negative number. 

$$
g(z) = z
$$

- **Sigmoid**: Used in the output layer for binary classification. It squashes the output between 0 and 1.

$$
g(z) = \frac{1}{1 + e^{-z}}
$$

- **ReLU**: Rectified Linear Unit. Used in hidden layers. When the input is negative (off range), the output is zero. It is faster to compute than the sigmoid function. Gradient descent is faster.

$$
g(z) = \max(0, z)
$$

### Multiclass Classification

Y can take more than two possible values. The output layer has one neuron for each class. The activation function is softmax, which squashes the output between 0 and 1 and normalizes the output so that the sum of the outputs is 1.
The output of neuron is the predicted probability for the class.

![alt text](images/multiclass_classification.png)

#### Softmax Function

 The softmax function can be written:

$$
a_j = \frac{e^{z_j}}{ \sum_{k=1}^{N}{e^{z_k} }}
$$

- $a_j$: The output of the $j$-th neuron.
- $k$: The index of the output neuron.
- $N$: The number of output neurons.

The output $a$ is a vector of length N, so for softmax regression, you could also write:

```math
a(x) =
\begin{bmatrix}
P(y = 1 | x; w,b) \\
\vdots \\
P(y = N | x; w,b)
\end{bmatrix} 
=
\frac{1}{ \sum_{k=1}^{N}{e^{z_k} }}
\begin{bmatrix}
e^{z_1} \\
\vdots \\
e^{z_{N}}
\end{bmatrix}
```

The output of the neurons z is passed through the softmax function to get the predicted probabilities for each class.

$$
z_1^{[3]} = \vec{w}_1^{[3]} \cdot \vec{a}^{[2]} + b_1^{[3]} 
$$

$$
z_2^{[3]} = \vec{w}_2^{[3]} \cdot \vec{a}^{[2]} + b_2^{[3]} 
$$

...

$$
z_n^{[3]} = \vec{w}_n^{[3]} \cdot \vec{a}^{[2]} + b_n^{[3]} 
$$

Probabilities for class 1:

$$
P(y=1|\vec{x}) = a_1^{[3]}
$$

$$
a_1^{[3]} = \frac{e^{z_1^{[3]}}}{e^{z_1^{[3]}} + ... + e^{z_n^{[3]}}}
$$


Probabilities for class n:

$$
P(y=n|\vec{x}) = a_n^{[3]}
$$

$$
a_n^{[3]} = \frac{e^{z_n^{[3]}}}{e^{z_1^{[3]}} + ... + e^{z_n^{[3]}}}
$$

- the output values sum to one
- the softmax spans all of the outputs. A change in z0 for example will change the values of a0-a3. Compare this to other activations such as ReLU or Sigmoid which have a single input and single output.

#### Cost Function for Multiclass Classification

The loss function associated with Softmax, the cross-entropy loss, is:

$$
\begin{equation}
  L(a,y)=\begin{cases}
    -log(a_1), & \text{if $y=1$}.\\
        &\vdots\\
     -log(a_N), & \text{if $y=N$}
  \end{cases}
\end{equation}
$$

![alt text](images/crosstropy_loss.png)

The cost function that covers all examples is:

```math
\begin{align}
J(\mathbf{w},b) = -\frac{1}{m} \left[ \sum_{i=1}^{m} \sum_{j=1}^{N}  1\left\{y^{(i)} == j\right\} \log \frac{e^{z^{(i)}_j}}{\sum_{k=1}^N e^{z^{(i)}_k} }\right]
\end{align}
```

Where $m$ is the number of examples, $N$ is the number of outputs. This is the average of all the losses.

Loss for a single example:

$$
loss(a_j) = -y_j \log(a_j)
$$

#### Multiclass Classification with Tensorflow

Model outputs the predicted probabilities for each class.

```python
model = Sequential([
  Dense(units=25, activation='relu'),
  Dense(units=15, activation='relu'),
  Dense(units=10, activation='softmax')
])

model.compile(loss=SparseCategoricalCrossEntropy())
model.fit(X,Y,epochs=100)

# Predict the probabilities for each class
probabilities = model(X)
```

A numerical more accourate way is to use a linear output layer and the softmax loss function. The model outputs the z1, z2, z3, ... zn. The probabilities are obtained by passing the logits through the softmax function.

```python
model = Sequential([
  Dense(units=25, activation='relu'),
  Dense(units=15, activation='relu'),
  Dense(units=10, activation='linear') # Model outputs z1, z2, z3, ... zn
])

# Softmax activation function in the loss function
model.compile(loss=SparseCategoricalCrossEntropy(from_logits=True))
model.fit(X,Y,epochs=100)

# the model does not output the probabilities, it outputs z1, z2, z3, ... zn
logits = model(X)

# to get the probabilities, pass the logits through the softmax function
f_x = tf.nn.softmax(logits)
```

### Multilable Classification

Multilabel classification is a classification task where each instance can belong to multiple classes. The output layer has one neuron for each class. The activation function is sigmoid, which squashes the output between 0 and 1.

Different from multiclass classification, where the sum of the probabilities is 1, in multilabel classification, the sum of the probabilities can be greater than 1.

![alt text](images/multilabel_classification.png)

### Adam Algorithm

Adam stands for Adaptive Moment Estimations. It adapts the learning rate during gradient descent. It became a defacto standard for training deep learning models.

```math
w := w - \alpha \frac{\partial}{\partial w} J(w, b) \\
b := b - \alpha \frac{\partial}{\partial b} J(w, b)
```

* If w or b keeps moving in the same direction it increases alpha, meaning it takes larger steps and it is faster. 
* If w or b keeps oscillating, it decreases alpha, meaning it takes smaller steps and it is more stable.

### Layers in a Neural Network

#### Dense Layer

Fully connected layer where each neuron is connected to all the neurons in the previous layer.

![alt text](images/dense_layer.png)

#### Convolutional Layer

Each neuron only looks at a part of the prevous layer output. Often used in image recognition where each neuron looks at a small part of the image. Computations are reduced and is more efficient.

![alt text](images/convolutional_layer.png)

## Evaluating the Model

If the model makes large errors in predictions (underfitting, high bias), make the model more complex:

- Try getting additional features
- Try adding polynomial features
- Try decreasing 𝜆

If the model fits the data to well (overfitting, high variance), make the model simpler:

- Get more training examples
- Try smaller sets of features
- Try increasing 𝜆

### The Problem of Overfitting and Underfitting

- A model with high bias does not fit the training set well. It is called underfitting.
- If it has a high variance, it fits the training set too well and does not generalize to new data. It is called overfitting.

Regression ![alt text](images/overfitting_regression.png)

Classification ![alt text](images/overfitting_classification.png)

### Regularization to Reduce Overfitting

- Collect more training data is the best solution.
- Select only the most important features based on intuition or domain knowledge.
- Reduce the weight of the less important features.

Regularization adds a penalty term to the cost function to reduce the complexity of the model. It discourages the weights from becoming too large, which can lead to overfitting.

L2 Regularization (Ridge)

$$
J(\vec{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{\vec{w}b}(\vec{x}^{(i)}) - y^{(i)} \right)^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2
$$

- $\lambda$ is the regularization parameter that controls the strength of the regularization.
- m is the number of training examples and n the number of features or weights.
- i refers to the training examples and j to the features or weights.
- if lambda is zero, the regularization term has no effect.
- the parameter $b$ is not regularized but can be included in the regularization term as well.
- higher order features usually have larger weights, those features typically end up being more penalized by the regularization term.

Gradient Descent with L2 Regularization:

$$
\vec{w} := \vec{w} - \alpha \left( \frac{1}{m} \sum_{i=1}^{m} \left( f_{\vec{w}b}(\vec{x}^{(i)}) - y^{(i)} \right) \vec{x}^{(i)} + \frac{\lambda}{m} \vec{w} \right)
$$

![alt text](images/cost_function_with_regularization.png)

The cost function (red line) increases the overall cost for large values of $w$.

#### Neural network regularization

```python
lam = 0.01

layer_1 = Dense(units=25, activation="relu", kernel_regularizer=L2(lam))
layer_2 = Dense(units=15, activation="relu", kernel_regularizer=L2(lam))
layer_3 = Dense(units=1, activation="sigmoid", kernel_regularizer=L2(lam))

model = Sequential([layer_1, layer_2, layer_3])
```


### Train test prodcedure for linear regression
Split the data into training $\vec{x}_{\text{train}}$ and test $\vec{x}_{\text{test}}$ sets. E.g. 70% training and 30% test.
The training set is used to train the model, and the test set is used to evaluate the model's performance on new, unseen data.

The parameters are fit by minimizing the cost function:
```math
J(\vec{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{\vec{w}b}(\vec{x}^{(i)}) - y^{(i)} \right)^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2
```

Compute the test error (MSE):
```math
J_{\text{test}}(\vec{w}, b) = \frac{1}{2m_{\text{test}}} \sum_{i=1}^{m_{\text{test}}} \left( f_{\vec{w}b}(\vec{x}^{(i)}_{\text{test}}) - y^{(i)}_{\text{test}} \right)^2
```

Compute the training error (MSE):
```math
J_{\text{train}}(\vec{w}, b) = \frac{1}{2m_{\text{train}}} \sum_{i=1}^{m_{\text{train}}} \left( f_{\vec{w}b}(\vec{x}^{(i)}_{\text{train}}) - y^{(i)}_{\text{train}} \right)^2
```

- If the model is overfitting, the training error will be low and the test error will be hight.
- If the model is underfitting, both the training and test error will be high.

### Train test prodcedure for logistic regression

Binary classification with logistic regression:
```math
\hat{y} = 
\begin{cases} 
1 & \text{if } f_{w,b}(x^{(i)}) \geq 0.5 \\
0 & \text{if } f_{w,b}(x^{(i)}) < 0.5 
\end{cases}
```
Split the data into training $\vec{x}_{\text{train}}$ and test $\vec{x}_{\text{test}}$ sets. E.g. 70% training and 30% test. Count the misclassified examples where the predicted value $\hat{y} \neq y$.

Fraction of the test set and the freaction of the train set that the algorithm has misclassified:

```math
J_{\text{test}}(w, b) = \frac{\text{Number of misclassified examples in the test set}}{\text{Total number of examples in the test set}}
```

```math
J_{\text{train}}(w, b) = \frac{\text{Number of misclassified examples in the train set}}{\text{Total number of examples in the train set}}
```

### Cross-Validation

Cross-validation is used to estimate how well the model will generalize to new, unseen data.

Split the data into training $\vec{x}_{\text{train}}$ and test $\vec{x}_{\text{test}}$ and validation $\vec{x}_{\text{cv}}$ sets. E.g. 60% training, 20% test, and 20% cross validation, sometimes also called the validation or development set.

Training error (MSE):
```math
J_{\text{train}}(\vec{w}, b) = \frac{1}{2m_{\text{train}}} \sum_{i=1}^{m_{\text{train}}} \left( f_{\vec{w},b}(\vec{x}_{\text{train}}^{(i)}) - y_{\text{train}}^{(i)} \right)^2
```

Test error (MSE):
```math
J_{\text{test}}(\vec{w}, b) = \frac{1}{2m_{\text{test}}} \sum_{i=1}^{m_{\text{test}}} \left( f_{\vec{w},b}(\vec{x}_{\text{test}}^{(i)}) - y_{\text{test}}^{(i)} \right)^2
```

Cross validation error (MSE) (also called validation error, dev error):
```math
J_{\text{cv}}(\vec{w}, b) = \frac{1}{2m_{\text{cv}}} \sum_{i=1}^{m_{\text{cv}}} \left( f_{\vec{w},b}(\vec{x}_{\text{cv}}^{(i)}) - y_{\text{cv}}^{(i)} \right)^2
```

Given some models with different degrees of polynomial features:
```math
\begin{align}
  f_{w,b}(x) &= w_1 x_1 + b \\
  f_{w,b}(x) &= w_1 x_1 + w_2 x_2 + b \\
  f_{w,b}(x) &= w_1 x_1 + w_2 x_2 + w_3 x_3 + b
\end{align}
```

1. Fit the parameters of each model with the training set $\vec{x}_{\text{train}}$
2. Choose the model (degree of polynomial) with the lowest cross-validation error $J_{\text{cv}}(w, b)$ based on the validation set $\vec{x}_{\text{cv}}$
3. Estimate the generalization error $J_{\text{test}}(\vec{w}, b)$ on this model with the test set $\vec{x}_{\text{test}}$.

This way, step 1 and 2 are used to select the model parameters. Step 1 chooses w and b and step 2 choose the degree of the polynomial. The test set is not involved in this process and can then be used to estimate a fair generalization error (unseen data) of the model.

- For binary classification $J_{\text{cv}}(w, b)$ is the fraction of misclassified examples. 
- Cross-validation is also used to choose layers and neurons in a neural network.
- Important to note, when using z-score feature nomalization the mean and standard deviation from the training set is also used to normalize the test and cross-validation sets.

#### K-Fold Cross-Validation

Split the data into k equal-sized folds. Train the model k times, each time using a different fold as the validation set and the remaining k-1 folds as the training set.

<img src="images/kfold_krossvalidation.png" height="300" />

- The cross-validation error is the average of the k cross-validation errors.
- K represents the number of folds.
- Choosing k = 5 or k = 10 is common. Higher values of k are more computationally expensive.
- Shuffle the data before splitting it into folds. This breaks any inherent order in the data. If the order is important, e.g. time series data, it should not be shuffled.

#### R2 Score

For regression, the R2 score is a measure of how well the model fits the data. 

```math
R^2 = 1− \frac{\text{Total Variation}}{\text{Unexplained Variation}}
```

```math
R^2 = 1− \frac{\text{SSres}}{\text{SStot}}
```

```math
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
```

- $y_i$ : Actual values.
- $\hat{y}_i$ : Predicted values.
- $\bar{y}$ : Mean of the actual values.

Interpretation:
- $R^2$ = 1 : Perfect fit. The model explains all the variability in the data.
- $R^2$ = 0 : The model does not explain any variability in the data, equivalent to predicting the mean of the target variable for all inputs.
- $R^2$ < 0 : The model performs worse than a simple mean prediction.

### Diagnosing Bias and Variance

High bias (underfit)
- $J_{\text{train}}$ will be high
- $J_{\text{train}} \approx J_{\text{cv}}$

High variance (overfit)
- $J_{\text{cv}} \gg J_{\text{train}}$
- $J_{\text{train}}$ may be low

High bias and high variance
- $J_{\text{train}}$ will be high
- $J_{\text{cv}} \gg J_{\text{train}}$

Example for a polynomial regression model. First both the training and cross-validation error are high. The model is underfitting. Then the training error decreases as the degree of polynomial increases. The cross-validation cost first decreases as well, but then increases again. The model is overfitting. Best fit is when the cross-validation error is at minimum (4 degree).

![alt text](images/degree_vs_cost.png)

Large $\lambda$ e.g. 10000
- High bias and low variance
- The model is underfitting
- w1, w2, w3, ... wn are close to zero

Small $\lambda$ e.g. 0.0001
- High variance and low bias ($\lambda$ has no effect)
- The model is overfitting
- w1, w2, w3, ... wn are large 

Choosing the regularization parameter $\lambda$ with cross-validation:

1. try a range of $\lambda$ values e.g. 0, 0.001, 0.01, 0.1, 1, 10, 100
2. fit the model with the training set $\vec{x}_{\text{train}}$
3. compute the cross-validation error $J_{\text{cv}}(w, b)$ with the cross-validation set $\vec{x}_{\text{cv}}$
4. Pick the $\lambda$ with the lowest cross-validation error
5. Estimate the generalization error $J_{\text{test}}(\vec{w}, b)$ with the test set $\vec{x}_{\text{test}}$

Bias and variance as a function of regularization parameter $\lambda$. With a small $\lambda$ the model has high variance, it is overfitting. With a large $\lambda$ the model has high bias and low variance, it is underfitting.

<img src="images/lambda_vs_cost.png" height="400" />

### Baseline level of performance

The baseline performance is the performance of a simple model or a human expert. It is used to compare the performance of the machine learning model. E.g. speak recognition model with 10.6% human error.

| Performance                                 | high variance | high bias | bigh bias and high variance|
|---------------------------------------------|---------------|-----------|----------------------------|
| Baseline performance (human)                | 10.6%         | 10.6%     | 10.6%                      |
| Training error $J_{\text{train}}(w, b)$     | 10.8%         | 15.0%     | 15.0%                      |
| Cross validation error $J_{\text{cv}}(w, b)$| 14.8%         | 15.5%     | 19.7%                      |

- **high variance**: difference between baseline and training error is low, but high between training and cross-validation error.
- **high bias**: difference between baseline and training error is high, but low between training and cross-validation error.
- **high bias and high variance**: difference is high between both baseline and training error and training and cross-validation error.

### Learning Curves

Learning curves are used to diagnose bias and variance, by plotting the training and cross-validation error as a function of the training set size.

Jtrain is low for small training set sizes, e.g. with only two data points the model can fit the data perfectly. The training error increases as the training set size increases.

Jcv is high for small training set sizes, the model does not generalize well. The cross-validation error decreases as the training set size increases.

<img src="images/training_set_size_vs_error.png" height="300" />

The curve flattens out. This shows that if a learining algorithm has high bias, getting more training data will not help but for high variance, getting more training could possibly be helpful.

### Bias and Variance in Neural Networks

Large neural networks are low bias machines, they fit complex functions very well. If Jtrain is high try a larger network. If Jcv is high as well try to get more training data.

A larger neural network will usually do as well or better than a smaller one so long as regularization is chosen correctly. But larger networks are more computationally expensive.

### Precision and Recall

In a skews dataset, e.g. 99% of the data is class 0 and 1% is class 1, accuracy is not a good metric. A model that always predicts class 0 will have an accuracy of 99%. Precision and recall are better metrics.

E.g. A classifer for a rare disease. The model should have a high recall, it should not miss any positive cases. The precision should also be high, it should not predict a positive case when it is not.

| Actual \ Predicted | Positive | Negative |
|--------------------|----------|----------|
| Positive           | TP       | FN       |
| Negative           | FP       | TN       |

- **Precision**: The ratio of correctly predicted positive **observations** to the total predicted positives.
  ```math
  \text{Precision} = \frac{TP}{TP + FP}
  ```

  E.g. of all patients that tested positive, how many actually have the disease. If false positive is high, we predict a lot of patients that do not have the disease.

  ```math
  \text{Precision} = \frac{15}{15 + 5} = 0.75
  ```

  The precision of 0.75, means that 75% of the patients that tested positive actually have the disease.

- **Recall (Sensitivity)**: The ratio of correctly predicted positive observations to all observations in the actual class.
  ```math
  \text{Recall} = \frac{TP}{TP + FN}
  ```

  E.g. of all patients that have the disease, how many did we correctly predict. If fals negative is high, we miss a lot of patients that have the disease.

  ```math
  \text{Recall} = \frac{15}{15 + 10} = 0.6
  ```

  The recall of 0.6, means that 60% of the patients that have the disease were correctly predicted.

- **Accuracy**: The ratio of correctly predicted observations to the total observations.
  ```math
  \text{Accuracy} = \frac{TP + TN}{TOTAL} = \frac{TP + TN}{TP + TN + FP + FN}
  ```

  E.g. of all patients, how many did we correctly predict.

  ```math
  \text{Accuracy} = \frac{15 + 5}{15 + 5 + 10 + 20} = 0.5
  ```

  The accuracy of 0.5, means that 50% of the patients were correctly predicted. Accuracy pimarily measures the number of correct predictions and is not a good metric for skewed datasets.

#### Trade off between precision and recall

Logist regression:

```math
\hat{y} = \begin{cases} 1 & \text{if } P(y=1|\vec{x}) \geq 0.5 \\ 0 & \text{otherwise} \end{cases}
```

If the threshold is increased to e.g. 0.7, the precision will increase but we miss some positive cases, the recall will decrease.

The precision-recall curve shows the trade off between precision and recall for different thresholds. The threshold for an algorithm is often chosen manually based on the requirements of the problem.

![alt text](images/precision_recall_curve.png)

#### F1 Score 

The F1 score combines precision and recall into a single metric. It is the harmonic mean of precision and recall. It is used when the classes are imbalanced.

```math
\text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
```

It can be used to compare different models. A model with a high F1 score has a good balance between precision and recall. 

| Algorithm  | Precision (P) | Recall (R) | F1 Score |
|------------|----------------|------------|----------|
| Algorithm 1| 0.5            | 0.4        | 0.44     |
| Algorithm 2| 0.7            | 0.1        | 0.18     |
| Algorithm 3| 0.02           | 1.0        | 0.04     |

Example: Algorithm 2 has a high precision but a low recall, algorithm 3 has a high recall but a low precision. Algorithm 1 has a good balance between precision and recall.

## Machine Learining Development Process

Iterative process to develop a machine learning model:

1. Choose architecture (model, data, features)
2. Train the model
3. Diagnostics (bias, variance and error analysis)

### Data Augmentation

Data augmentation is a technique used to increase the size of the training set by applying transformations to the data. It is used to reduce overfitting and improve the generalization of the model. E.g. flipping, rotating, scaling, cropping, and changing the brightness of images.

### Data Synthesis

Data synthesis is a technique using artificial data inputs to create new data. It is used when the training data is limited. E.g. generating new images of letters and numbers for OCR.

### Transfer Learning

Transfer learning is a technique where a model trained on one task with large amount of data is used as a starting point (supervised pretraining). This model and its parameters is then copied and used as a starting point for a new task. The model is then fine-tuned on the new task. 

Pre-trained models are available for image recognition, speech recognition, and natural language processing.

You can use an open source pre-trained neuronal network and just train the last layer for your specific task or use it as a starting point and train all parameters of all layers.

## Decision Trees

Decision trees are used for classification and regression. They are easy to interpret and visualize. They can handle both numerical and categorical data.

A decision tree is a tree where each node represents a feature (ear shape, face shape, whiskers), each branch represents a decision, and each leaf represents an outcome.

![alt text](images/decision_tree.png)

There are multiple possible decision trees for a dataset. The job of the learning algorithm is to find the best tree that fits the training data and generalizes well to new, unseen data.

**Decision 1**: How to split the data at each node. 
- The goal is to maximize purity (or minimize impurity) at each node. Ideally, if there where only two classes, one class would be in one leaf and the other class in the other leaf.

**Decision 2**: When to stop splitting the data. 
- When a node is considered pure, 100% of the data belongs to one class
- When the tree reaches a maximum depth.
- When improvements in purity score are below a certain threshold.
- When then number of samples in a node is below a certain threshold.

### Entropy as a measure of impurity

**Entropy** is a measure of impurity in a dataset. 

**Purity** is the opposite of impurity. A dataset is pure if all the data belongs to the same class. A dataset is impure if the data is evenly distributed among the classes.

- $p_1$ is the proportion of examples in class. E.g. fraction of examples that are cats.
- $p_0$ is the proportion of examples that or not in the class. E.g. fraction of examples that are not cats.

```math
p_0 = 1 - p_1
```

```math
H(p_1) = -p_1 \log_2(p_1) - p_0 \log_2(p_0)
```

![alt text](images/entropy_function.png)

Examples:

- 80% of the examples are cats and 20% are not cats: p1 = 0.8 and H(p1) = 0.72
- 50% of the examples are cats and 50% are not cats: p1 = 0.5 and H(p1) = 1.0
- If either all the examples are cats or none of the examples are cats, the entropy is 0.

### Information Gain

**Information gain** is the reduction in entropy or impurity. The goal is to reduce the entropy at each node.

```math
\text{Information Gain} = \text{Entropy(parent)} - \text{Weighted Average Entropy(children)}
```

![alt text](images/information_gain.png)

- $p_1^{root}$ is the proportion of positive examples at the root node.
- $p_1^{left}$ and $p_1^{right}$ are the proportions of examples in class 1 at the left and right child nodes. E.g. fraction of examples that are cats.
- $w^{left}$ and $w^{right}$ are the weights of the left and right child nodes. E.g. $w^{left}$ is the samples in the left child node divided by the total number of samples from the parent node.
- $H(p_1)$ is the entropy at the node.

```math
\text{Information Gain} = H(p_1^{root}) - (w^{left} H(p_1^{left}) + w^{right} H(p_1^{right}))
```

### Decision Tree Learning Algorithm (Recursive Splitting)

1. Start with all examples at the root node
2. Calculate information gain for all possible features, and pick the one with the highest information gain
3. Split dataset according to selected feature, and create left and right branches of the tree
4. Keep repeating the splitting process until a stopping criteria is met:
      - When a node is 100% one class
      - When splitting a node will result in the tree exceeding a maximum depth
      - Information gain from additional splits is less than threshold
      - When number of examples in a node is below a threshold

After deciding on the root node, the algorithm is recursively applied to each child node. Each child node repeats the process of selecting the feature on a subset of the data from the parent node.

### Features with Multiple Classes

Features with multiple classes create multiple branches in the tree. E.g. ear shape (pointy, round, floppy) creates three sub branches.

| Ear shape | Face shape | Whiskers | Cat |
|-----------|------------|----------|-----|
| Pointy    | Round      | Present  | 1   |
| Floppy    | Round      | Absent   | 0   |
| Oval      | Round      | Absent   | 1   |
| Floppy    | Not round  | Absent   | 0   |

#### One-Hot Encoding

One-hot encoding converts a feature with multiple classes into multiple binary features.

Example for ear shape with three classes (pointy, round, floppy):

| Ear shape | Face shape | Whiskers | Pointy | Floppy | Oval   | Cat |
|-----------|------------|----------|--------|--------|--------|-----|
| Pointy    | Round      | Present  | 1      | 0      | 0      | 1   |
| Floppy    | Round      | Absent   | 0      | 1      | 0      | 0   | 
| Oval      | Round      | Absent   | 0      | 0      | 1      | 1   |
| Floppy    | Not round  | Absent   | 0      | 1      | 0      | 0   |

If a categorical feature has k classes, one-hot encoding will create k binary features.

With one-hot encoding, the decision tree algorithm can handle features with multiple classes with a binary split.

### Splitting Continuous Variables

Continuous features can have any value in a range. The decision tree algorithm tries different split points to find the best split with the highest information gain.

![alt text](images/information_gain_continuous.png)

Split points are typically choosen by sorting the unique values and then calculate the average of two consecutive values.

```math
x_{\text{split}} = \frac{x_i + x_{i+1}}{2}, \quad \text{for } i \in \{1, 2, \ldots, n-1\}.
```

### Regression with Decision Trees

If the value to predict is continuous, the decision tree is used for regression. The process for the learning algorithm is the same as for classification, but the impurity measure is different.

In the example instead of trying to predict cat or not cat, the decision tree tries to predict the weight of the animal. For unseen data, the decision tree will predict the weight of the animal based on the average weight of the animals in the leaf node.

![alt text](images/decision_tree_regression.png)

For regression, instead of entropy, the variance is used as a measure of impurity. The goal is to minimize the variance of the target variable at each node.

![alt text](images/decision_tree_splint_regression.png)

The data at a node is split to minimize the variance of the target variable. The variance is calculated as the average of the squared differences between the target variable and the mean of the target variable.

```math
\text{Variance} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \bar{y})^2
```

The split enusres that each leaf node has a lower variance than the parent node.

```math
\text{Variance}_{parent} > \text{Weighted Variance}_{children}
```

Weighted variance of the children is the average of the variance of the left and right child nodes. $n$ is the number of examples in the parent node, $n_{\text{left}}$ and $n_{\text{right}}$ are the number of examples in the left and right child nodes.

```math
\text{Weighted Variance}_{childeren} = \frac{n_{\text{left}}}{n} \text{Variance}_{left} + \frac{n_{\text{right}}}{n} \text{Variance}_{right}
```

Calculate the variance reduction:
  
```math
\text{Variance Reduction} = \text{Variance}_{parent} - \text{Weighted Variance}_{children}
```

### Tree Ensembles

One weakness of using a single decision tree is that it can be sensitive to small changes in the training data. Small changes in the data could lead to a completely different tree. One solution is to use multiple decision trees and combine their predictions.

![alt text](images/decision_tree_ensemble.png)

Hyperparameters to tune:
- Number of trees $B$
- Number of features $k$
- Maximum depth of the tree
- Minimum number of samples required to split a node
- Minimum number of samples required at each leaf node

#### Bagging (Bootstrap Aggregating)

Bagging uses sampling with replacement (bootstrap) to create multiple datasets. Each dataset contains a subset of the examples and features. And example can be used multiple times in the dataset. Each dataset is used to train a another decision tree.

- Given training set of size 𝑚
  - For $b$ = 1 to $B$:
    - Use sampling with replacement to create a new training set of size 𝑚
    - Train a decision tree on the new dataset

$B$ is typically in the range of 64 to 128.

#### Random Forest Algorithm

In addition to bagging, Random Forest introduces randomness. At each node, when choosing a feature to use to split, if $n$ features are available, pick a random subset of $k < n$ features and allow the algorithm to only choose from that subset of features.

- Given training set of size 𝑚
  - For $b$ = 1 to $B$:
    - Use sampling with replacement to create a new training set of size 𝑚
    - Train a decision tree on the new dataset, but at each node:
      - Randomly select a subset of features of size $k$.
      - Choose the best feature to split on from the subset of features

$k$ is typically the square root of the number of features. Used when the number of features is large.

#### Gradient Boosting
The algorithm iteratively adds trees to correct the errors from the previous trees, each new tree aims to correct the errors made by the previous ones. Instead of sampling with replacement, the algorithm samples the examples with weights. Examples that are misclassified have a higher weight.

- Start with a constant model that predicts the mean of the target variable for regression tasks or the most frequent class for classification tasks.
- For $b$ = 1 to $B$:
  - Calculate the difference between the actual target values and the current model’s predictions. These differences are known as residuals.
  - Train a new decision tree to predict these residuals. This tree focuses on the errors made by the previous trees
  - Add the new tree to the existing ensemble with a scaling factor (learning rate) to control the contribution of the new tree.

The final model is the sum of the initial model and all the trees added during the iterations.

Details of the Gradient Boosting Algorithm for Regression:

1. Initial Prediction: 

```math
F_0(x) = \text{mean}(y)
```
2. Compute the residuals $r_i$ for each data point (difference between the actual target value and the current model’s prediction): 
```math
r_i = - \left[ \frac{\partial L(y_i, F_{m-1}(x_i))}{\partial F_{m-1}(x_i)} \right]
```
For squared error loss function, it is simply the difference between actual target value and the current model’s prediction:
```math
r_i = y_i - F_0(x_i)
```
3. Train  weak learner (decision tree) $h_1(x)$ to predict the residuals $r_i$ instead of the target $y$ using the input features $x_i$
```math
h_1(x) = \text{argmin}_{h} \sum_{i=1}^{m} (r_i - h(x_i))^2
```
4. Update the model $F_1(x)$ with the new tree $h_1(x)$ and a learning rate $\alpha$ (e.g. 0.1):
```math
F_1(x) = F_0(x) + \alpha h_1(x)
```
5. Compute the residuals $r_i$ for the new model:
```math
r_i = y_i - F_1(x_i)
```
6. Continue training new trees (step 3) and updating the model until the residuals are small or a stopping criteria is met.

#### XGBoost

XGBoost is a popular implementation of the gradient boosting algorithm. XGBoost builds upon the gradient boosting framework, which constructs an ensemble of decision trees sequentially. Each new tree aims to correct the errors made by the previous ones.

#### Decision Trees vs Neural Networks

Decision Trees:
- Work well on tabular (structured) data
- Not recommended for unstructured data (images, audio, text)
- Decision tree models are fast to train and easy to interpret

Neural Networks:
- Work well on all types of data, including tabular (structured) and unstructured data (images, audio, text)
- Maybe slow to train and require a lot of data
- Work with transfer learning and pre-trained models
- When building a system of multiple models working together, it might be easier to string together multiple neural networks

## Clustering

Clustering is an unsupervised learning technique used to group similar data points together. The goal is to find groups of data points that are similar to each other and dissimilar to data points in other groups.

### K-Means Clustering

K-means partitions the data points into k clusters. K needs to be specified by the user. The algorithm assigns each data point to the cluster with the nearest centroid. The centroid is the mean of all the data points in the cluster.

1. Randomly initialize $k$ cluster centroids $\mu_1, \mu_2, \ldots, \mu_k$
2. Assign each data point to the nearest centroid
```math
c^{(i)} = \text{argmin}_k ||x^{(i)} - \mu_k||^2
```
3. Update the centroids by calculating the mean of all the data points in the cluster
```math
\mu_k = \frac{1}{|C_k|} \sum_{i \in C_k} x^{(i)}
```
4. Repeat steps 2 and 3 until the centroids do not change or a stopping criteria is met

Cost function for a single cluster:
```math
J(c^{(1)}, \ldots, c^{(m)}, \mu_1, \ldots, \mu_k) = \frac{1}{m} \sum_{i=1}^{m} ||x^{(i)} - \mu_{c^{(i)}}||^2
```
Cost function for all clusters:
```math
J = \sum_{i=1}^k \sum_{x \in C_k} \| x - \mu_{c^{(i)}} \|^2
```

- $c^{(i)}$ index of cluster (1,2..,k) to which example $x^{(i)}$ is currently assigned
- $C_k$ is the set of data points assigned to cluster $k$
- $|C_k|$ is the number of data points assigned to cluster $k$
- $u_k$ is the centroid of cluster $k$
- $u_{c^{(i)}}$ is the centroid of the cluster to which example $x^{(i)}$ is currently assigned
- $m$ is the number of data points

If a cluster has no data points assigned, we can either remove the cluster or reinitialize the centroids.

![alt text](images/k_means_clustering.png)

**Random initialization**:
1. choose $k$ < m
2. Randomly select $k$ training examples
3. Set $\mu_1, \mu_2, \ldots, \mu_k$ equal to these $k$ examples

The cost function should go down with each iteration. The algorithm may converge to a local minimum, so it is recommended to run the algorithm multiple times (50-100) with different initializations and pick the set of clusters with the lowest cost (distortion).

Example for k = 3 clusters with different initializations and different local minima:

![alt text](images/k_means_initialization.png)

**Choosing the number of clusters $k$:**
- The elbow method: plot the cost function as a function of the number of clusters. The cost function will decrease as the number of clusters increases. The elbow point is the point where the cost function starts to decrease more slowly.
- It is often choosen based on domain knowledge.

## Anomaly Detection

Anomaly detection is used to identify data points that are significantly different from the rest of the data. Anomalies are also called outliers, novelties, noise, deviations, and exceptions.

Use cases:
- Fraud detection
  - Features of users activities, e.g. time of day, location, amount of transaction
  - Model $p(x)$ from the data
  - Identify data points with low probability $p(x) < \epsilon$
- Manufacturing
  - Features of machines, e.g. temperature, pressure, vibration
- Intrusion detection
  - Features of network traffic, e.g. number of packets, time of day, source IP address

### Gaussian Distribution

Probability is determined by the Gaussion (normal) distribution with mean $\mu$ and variance $\sigma^2$.
```math
p(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi}\sigma} \exp \left( - \frac{(x - \mu)^2}{2\sigma^2} \right)
```
- $x$ is the data point
- $\mu$ is the mean of the data
- $\sigma^2$ is the variance of the data

Parameters $\mu$ and $\sigma^2$ are estimated from the data:
```math
\mu = \frac{1}{m} \sum_{i=1}^{m} x^{(i)}
```
```math
\sigma^2 = \frac{1}{m} \sum_{i=1}^{m} (x^{(i)} - \mu)^2
```

### Density Estimation

Dennsity estimation is used to model the distribution of the data. The goal is to find data that has low probability of occuring

```math
p(x_{\text{test}}) < \epsilon
```
- $p(x_{\text{test}})$ is the probability of the data point $x_{\text{test}}$
- $\epsilon$ is the threshold
  
**General Formula for Anomaly Detection**:
```math
p(x) = p(x_1; \mu_1, \sigma_1^2) \cdot p(x_2; \mu_2, \sigma_2^2) \cdot \ldots \cdot p(x_n; \mu_n, \sigma_n^2) = \prod_{j=1}^{n} p(x_j; \mu_j, \sigma_j^2)
```
- $x_i$ is the $i$-th feature of the data point $x$
- $\mu_i$ is the mean of the $i$-th feature
- $\sigma_i^2$ is the variance of the $i$-th feature
- $p(x_i; \mu_i, \sigma_i^2)$ is the probability density function of the $i$-th feature

The overall probability $p(x)$ is the product of the probabilities of each feature. Here we assume that the features are independent, which is not always the case but is a common assumption. If one of the features has a low probability, the overall probability will be low.

Example with two features $x_1$ and $x_2$:

![alt text](images/density_estimation.png)

### Anomaly Detection Evaluation

To evaluate the performance of the anomaly detection algorithm, we need labeled data.

Assume labeled data with: y = 0 for normal data and y = 1 for anomalies.

- **Training set:** $x^{(1)}, x^{(2)}, \ldots, x^{(m)}$ for all data with $y = 0$
- **Coss-validation set:** $x_{\text{cv}}^{(1)}, x_{\text{cv}}^{(2)}, \ldots, x_{\text{cv}}^{(m_{\text{cv}})}$ for all data with $y_{\text{cv}} = 0$ and $y_{\text{cv}} = 1$
- **Test set:** $x_{\text{test}}^{(1)}, x_{\text{test}}^{(2)}, \ldots, x_{\text{test}}^{(m_{\text{test}})}$ for all data with $y_{\text{test}} = 0$ and $y_{\text{test}} = 1$

Example with Airplane engine data (ratio good to faulty engines is very low):
- 10000 good engines (y = 0)
- 20 faulty engines (y = 1)

We add all the good engines to the training set fit the Gaussian distribution and calculate the mean and variance. We then use the cross-validation set to find the best threshold $\epsilon$.
- Training set: 6000 good engines
- Cross-validation set: 2000 good engines, 10 faulty engines
- Test set: 2000 good engines, 10 faulty engines

Event though we work with labeled data, it is still an unsupervised learning problem because we do not use the labels in the training process.

Evaluation Metrics on the cross-validation set:

```math
y = \begin{cases} 
1 & \text{if } p(x) < \epsilon \text{ (anomaly)} \\
0 & \text{if } p(x) \geq \epsilon \text{ (normal)}
\end{cases}
```

- True positive, false positive, true negative, false negative
- Precision/recall
- F1 score

### Anomaly Detection vs Supervised Learning

Anomaly detection:
- very small number of positive examples $y=1$ (anomalies) and large number of negative examples $y=0$. (0-20 examples is common)
- Many different types of anomalies are hard for an algorithm to learn. E.g. many different types of fraud or many different types of manufacturing defects. Future anomalies may look very different from the anomalies in the training set.

Supervised learning:
- large number of positive and negative examples
- Enough positives examples for the algorithm to learn. Future positive examples are likely to be similar to the positive examples in the training set.

### Feature Selection in Anomaly Detection

Correct feature selection is more important in anomaly detection than in supervised learning. Supervised learning algorithms can learn to ignore irrelevant features. Anomaly detection algorithms can be sensitive to irrelevant features.

Make sure features are more or less Gaussian distributed

Features that are not guassian distributed can be transformed with a log transformation or a square root transformation.
  
```math
x_i = \log(x_i + c)
```
```math
x_i = \sqrt{x_i} = x_i^{\frac{1}{2}}
```
The value of $c$ is choosen to avoid taking the log of zero. The value of $c$ is set experimentally.

![alt text](images/log-transformation.png)

All the transformations applied to the training set must also be applied to the cross-validation and test sets.

## Recommender Systems
Recommender systems are used to recommend items to users based on their preferences. They are used in e-commerce, social media, and streaming services.

### Collaborative Filtering
If we have ratings from multiple users and features for the items, we can use supervised learning algorithms (linear regression) to predict the rating of a user for an item.
All users collaborate to generate the rating set.

| Movie                  | Alice(1) | Bob(2) | Carol(3) | Dave(4) | x1 (romance) | x2 (action) |
|------------------------|----------|--------|----------|---------|--------------|-------------|
| Love at last           | 5        | 5      | 0        | 0       | 0.9          | 0           |
| Romance forever        | 5        | ?      | 0        | 0       | 1.0          | 0.01        |
| Cute puppies of love   | ?        | 4      | ?        | ?       | 0.99         | 0           |
| Nonstop car chases     | 0        | 0      | 4        | 4       | 0.1          | 1.0         |
| Swords vs. karate      | 0        | 0      | ?        | ?       | 0            | 0.9         |

- $n_u$ is the number of users
- $n_m$ is the number of movies
- $r(i, j) = 1$ if user $j$ has rated movie $i$. e.g. $r(1, 1) = 1$ and $r(2, 2) = 0$
- $y(i, j)$ is the rating given by user $j$ to movie $i$. e.g. $y(1, 1) = 5$ and $y(2, 4) = 0$
- $x^{(i)}$ is the feature vector for movie $i$, e.g. $x^{(1)} = \begin{bmatrix} 0.9 \\ 0 \end{bmatrix}$
- $m_j$ is the number of movies rated by user $j$

Predict the rating for moive Cute puppies of love for user Alice. Parameters $w$ and $b$ are learned from the data. $x$ is the feature vector for the movie. 
$w$ represents the taste of a user and has the same length as $x$.
```math
\text{rating} = w^{(1)} * x^{(3)} + b^{(1)} = \begin{bmatrix} 5 \\ 0 \end{bmatrix} * \begin{bmatrix} 0.99 \\ 0 \end{bmatrix} + 0 = 4.95
```

To learn the parameters $w^{(j)}$ and $b^{(j)}$ for a user $j$ we minimize the cost function:
```math
J(w^{(j)}, b^{(j)}) = \frac{1}{2} \sum_{i:r(i, j) = 1} \left( w^{(j)} * x^{(i)} + b^{(j)} - y^{(i, j)} \right)^2 + \frac{\lambda}{2} \sum_{k=1}^{n} (w_k^{(j)})^2 
```

- $y^{(i, j)}$ is the rating by user $j$ on movie $i$. We exculde the movies that have not been rated from the sum. $r(i, j) = 1$ if user $j$ has rated movie $i$.
- $\lambda$ is the regularization parameter
- $n$ is the number of features, e.g. $n = 2$ in the example

To learn all the parameters $w^{(j)}$ and $b^{(j)}$ for all users we minimize the cost function:
```math
J(w^{(1)}, b^{(1)}, \ldots, w^{(n_u)}, b^{(n_u)}) = \frac{1}{2} \sum_{j=1}^{n_u} \sum_{i:r(i, j) = 1} \left( w^{(j)} * x^{(i)} + b^{(j)} - y^{(i, j)} \right)^2 + \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^{n} (w_k^{(j)})^2 
```

- the vector $w^{(j)}$ is the parameter vector for user $j$ and has the same length as the feature vector $x^{(i)}$
- Each has one bias term $b^{(j)}$
- Parameters per user: $n + 1$. Toatal parameters for $n_u$ user and $n$ features: $n_u * (n + 1)$

#### Collaborative Filtering Algorithm

In the example above we used the features of the movies to predict the ratings of the users. 
In collaborative filtering we use the ratings of the users to learn the features of the movies if we don't have the features.

| Movie                  | Alice(1) | Bob(2) | Carol(3) | Dave(4) | x1 (romance) | x2 (action) |
|------------------------|----------|--------|----------|---------|--------------|-------------|
| Love at last           | 5        | 5      | 0        | 0       | ?            | ?           |
| Romance forever        | 5        | ?      | 0        | 0       | ?            | ?           |
| Cute puppies of love   | ?        | 4      | ?        | ?       | ?            | ?           |
| Nonstop car chases     | 0        | 0      | 4        | 4       | ?            | ?           |
| Swords vs. karate      | 0        | 0      | ?        | ?       | ?            | ?           |

Given $w^{(1)}, b^{(1)}, \ldots, w^{(n_u)}, b^{(n_u)}$ we can learn the features $x$ for the movies. We can then use the features to predict the ratings of the users.

To learn $x$ for a movie $i$ we minimize the cost function:
```math
J(x^{(i)}) = \frac{1}{2} \sum_{j:r(i, j) = 1} \left( w^{(j)} * x^{(i)} + b^{(j)} - y^{(i, j)} \right)^2 + \frac{\lambda}{2} \sum_{k=1}^{n} (x_k^{(i)})^2 
```

For all movies we minimize the cost function:
```math
J(x^{(1)}, \ldots, x^{(n_m)}) = \frac{1}{2} \sum_{i=1}^{n_m} \sum_{j:r(i, j) = 1} \left( w^{(j)} * x^{(i)} + b^{(j)} - y^{(i, j)} \right)^2 + \frac{\lambda}{2} \sum_{i=1}^{n_m} \sum_{k=1}^{n} (x_k^{(i)})^2 
```

- $n_m$ is the number of movies
- $n$ is the number of features, e.g. $n = 2$ in the example
- $y^{(i, j)}$ is the rating by user $j$ on movie $i$. We exculde the movies that have not been rated from the sum. $r(i, j) = 1$ if user $j$ has rated movie $i$.

We can combine the cost function to learn the parameters $w$, $b$ and the cost function to learn the features $x$ and minimize the cost function for all parameters:

```math
J(w, b, x) = \frac{1}{2} \sum_{(i, j):r(i, j) = 1} \left( w^{(j)} * x^{(i)} + b^{(j)} - y^{(i, j)} \right)^2 + \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^{n} (w_k^{(j)})^2 + \frac{\lambda}{2} \sum_{i=1}^{n_m} \sum_{k=1}^{n} (x_k^{(i)})^2 
```

We can use gradient descent or other optimization algorithms to minimize the cost.
```math
w_j^{(j)} := w_j^{(j)} - \alpha \frac{\partial}{\partial w_j^{(j)}} J(w, b, x)
```
```math
b^{(j)} := b^{(j)} - \alpha \frac{\partial}{\partial b^{(j)}} J(w, b, x)
```
```math
x_k^{(i)} := x_k^{(i)} - \alpha \frac{\partial}{\partial x_k^{(i)}} J(w, b, x)
```

#### Binary Labels

If we don't have ratings like 1 to 5, but have the information if a user likes a movie or not, we have so called binary labels.
The labels are 1 if the user likes the movie and 0 if the user does not like the movie.
Other examples are: click/no click, buy/no buy. 

For binary labels the predict is similar to logistic regression:
```math
g(z) = \frac{1}{1 + e^{-z}}
```
```math
y^{(i, j)}: f_{(w,b,x)}(x) = g(w^{j} * x^{(i)} + b^{(j)})
```
The loss for a single example is:
```math
L(f_{(w,b,x)}(x), y^{(i, j)}) = -y^{(i, j)} \log(f_{(w,b,x)}(x) - (1 - y^{(i, j)}) \log(1 - f_{(w,b,x)}(x)
```
The cost function is the sum of the loss for all examples:
```math
J(w, b, x) = \frac{1}{m} \sum_{i=1}^{m} L(f_{(w,b,x)}(x), y^{(i, j)}) + \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^{n} (w_k^{(j)})^2 + \frac{\lambda}{2} \sum_{i=1}^{n_m} \sum_{k=1}^{n} (x_k^{(i)})^2 
```

- $m$ is the number of examples
- $n_u$ is the number of users
- $n_m$ is the number of movies
- $n$ is the number of features
- $y^{(i, j)}$ is the label for the example $i$ and user $j$. $y^{(i, j)} = 1$ if the user likes the movie and $y^{(i, j)} = 0$ if the user does not like the movie.
- $f_{(w,b,x)}(x)$ is the prediction of the model
- $x$ is the feature vector for the movie

#### Mean Normalization

Mean normalization is used to normalize the ratings of the users. The mean of the ratings is subtracted from the ratings.
```math
y^{(i, j)} = y^{(i, j)} - \mu
```

If we have a user that has not rated any movies, the prediction for the user will be the mean of all the ratings.

Example:

Given 5 users $n$ and 5 movies $m$, where user 5 has not rated any movies:
```math
\begin{bmatrix}
5 & 5 & 0 & 0 & ? \\
5 & ? & ? & 0 & ? \\
? & 4 & 0 & ? & ? \\
0 & 0 & 5 & 4 & ? \\
0 & 0 & 5 & 0 & ?
\end{bmatrix}
```
The mean of the ratings for each movie is:
```math
\begin{bmatrix}
2.5 \\
2.5 \\
2 \\
2.25 \\
1.25 
\end{bmatrix}
```

then we subtract the mean from the ratings to get the normalized ratings:
```math
\begin{bmatrix}
2.5 & 2.5 & -2.5 & -2.5 & ? \\
2.5 & ? & ? & -2.5 & ? \\
? & 2 & -2 & ? & ? \\
-2.25 & -2.25 & 2.75 & 1.75 & ? \\
-1.25 & -1.25 & 3.75 & -1.25 & ?
\end{bmatrix}
```

The we can learn the ratings like in the previous examples. For users that have not rated any movies, the prediction will be the mean of the ratings.
We just need to add the mean to the prediction to get the actual rating.
```math
w^{(5)} * x^{(i)} + b^{(5)} + \mu_i
```

#### Finding Similar Items

Lets say user selects a movie $i$ and we want to recommend a similar movie $k$. 
We can use the features $x$ to find the similarity between the movies by calculating the distance between the feature vectors with the Euclidean distance.

```math
\text{similarity} = \sum_{j=1}^{n} (x_j^{(i)} - x_j^{(k)})^2 = ||x^{(i)} - x^{(k)}||^2
```

#### Collaborative Filtering in Tensorflow

Neural networks are not typically used for collaborative filtering. But we can user TensorFlow auto differentiation to learn the parameters $w$, $b$ and $x$.

Example for gradient descent:

```python
w = tf.Variable(3.0)
x = 1.0
y = 1.0 # target value
alpha = 0.01
iterations = 30

for iter in range(iterations):
  # Use TensorFlow’s Gradient tape to record the steps
  # used to compute the cost J, to enable auto differentiation.
  with tf.GradientTape() as tape:
    fwb = w*x
    costJ = (fwb - y)**2
  
  # Use the gradient tape to calculate the gradients
  # of the cost with respect to the parameter w.
  [dJdw] = tape.gradient( costJ, [w] )
  
  # Run one step of gradient descent by updating
  # the value of w to reduce the
  w.assign_add(-alpha * dJdw)
```

Example for Adam optimizer:

```python
# Instantiate an optimizer.
optimizer = keras.optimizers.Adam(learning_rate=1e-1)
iterations = 200

for iter in range(iterations):
  # Use TensorFlow’s GradientTape
  # to record the operations used to compute the cost
  with tf.GradientTape() as tape:
    # Compute the cost (forward pass is included in cost)
    cost_value = cofiCostFuncV(X, W, b, Ynorm, R, num_users, num_movies, lambda)
  
  # Use the gradient tape to automatically retrieve
  # the gradients of the trainable variables with respect to the loss
  grads = tape.gradient( cost_value, [X,W,b] )
  
  # Run one step of gradient descent by updating
  # the value of the variables to minimize the loss.
  optimizer.apply_gradients( zip(grads, [X,W,b]) )
```

#### Limitations of Collaborative Filtering

- Cold start problem: If a new user or movie is added, we don't have any ratings for the user or movie. We can't make any recommendations.
- Show something reasonable to new users who have rated few items.
- Meta informations such as genres, actors, directors, etc. or information about the user like age or country are not used in collaborative filtering.


### Content-Based Filtering

Collaborative filterind:
- Recommend items based on rating of users who gave similar ratings as you.

Content-based filtering:
- Recommend items based on the features of user and items (content).

User features:
- age
- gender
- country
- movies watched
- average rating per genere

Movie features:
- genere
- year
- actors
- reviews

Notation:
- $r(i, j)$ = 1 if user j has rated movie i
- $y(i, j)$ = rating by user j on movie i
- $x_u^{(j)}$ = feature vector for user j
- $x_m^{(i)}$ = feature vector for movie i

$x_u^{(j)}$ and $x_m^{(i)}$ are vectors with difference lengths. 
To predict the rating of user j on movie i we can use the dot product of the vectors $v_u^{(j)}$ and $v_m^{(i)}$:
These vectors have the same length and are computed from the features $x_u^{(j)}$ and $x_m^{(i)}$.

```math
\text{prediction} = v_u^{(j)} \cdot v_m^{(i)}
```

To predict the probability of user $j$ liking movie $i$ we can use the sigmoid function.
```math
p = g(v_u^{(j)} \cdot v_m^{(i)})
```
```math
p = \frac{1}{1 + e^{-v_u^{(j)} \cdot v_m^{(i)}}}
```

We can use a neural network to learn the vector $v_u^{(j)}$  that describes the user $j$ 
and a neural network to learn the vector $v_m^{(i)}$ that describes the movie $i$.
The two networks can have different architectures, but the ouput layer needs to have the same length.

<img src="images/recommender_neural_network.png" height="400" />

The parameters of the networks are trained together to minimize the cost function. 
Same as collaborative we only use the examples where the user has rated the movie $r(i, j) = 1$.
```math
J(v_u^{(1)}, \ldots, v_u^{(n_u)}, v_m^{(1)}, \ldots, v_m^{(n_m)}) = \frac{1}{2} \sum_{(i, j):r(i, j) = 1} \left( v_u^{(j)} \cdot v_m^{(i)} - y^{(i, j)} \right)^2 + \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^{n} (v_k^{(j)})^2 + \frac{\lambda}{2} \sum_{i=1}^{n_m} \sum_{k=1}^{n} (v_k^{(i)})^2 
```

To finde similar movies we can use the euclidean distance between the vectors $v_m^{(i)}$ and $v_m^{(k)}$.
```math
\text{similarity} = ||v_m^{(i)} - v_m^{(k)}||^2
```

#### Retrival and Ranking

To efficiently recommend items to a user, it is not necesary to compute the ranking for all items when the user logs in.
The vector $v_m^{(i)}$ for all movies can be precomputed and stored in a database.

Retrival:
 1.  For each of the last 10 movies watched by the user, find 10 most similar movies: $||v_m^{(j)} - v_m^{(k)}||^2$
 2.  For most viewed 3 genres, find the top 10 movies
 3.  Top 20 movies in the country
 4.  Combine retrieved items into list, removing duplicates and items already watched/purchased

Ranking:
1. Take list of movies retrieved and rank using the trained model: $v_u^{(j)} \cdot v_m^{(i)}$


## Reinforcement Learning

Reinforcement learning is used to teach a software agent  how to make decisions. The goal is to learn a policy that maps states to actions that maximizes the reward.

- $s_g$ is the current state
- $a_g$ is the current action
- $R(s)_g$ is the reward for state $s$
- $s'$ is the next state
- $a'$ the action in the next state
- $\gamma$ is the discount factor (0 to 1) usually 0.9 or close to 1

The return $G_t$ is the sum of the rewards from time $t$ to the end of the episode.
```math
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
```
The return depends on the rewards you get and the reward depends on the actions you take.
The discount factor $\gamma$ is used to give more weight to immediate rewards.

The policy $\pi$ is a function used to select actions. It mapps from states to actions.
```math
\pi(a|s) = P(a|s)
```

- $P(a|s)$ is the probability of taking action $a$ in state $s$
- $\pi(a|s)$ is the policy that selects the action $a$ in state $s$

### Markov Decision Process

In a markov decision process (MDP) the future is only determined by the current state ant not how you got in this state.

<img src="images/mdp.png" height="300" />

Concepts:

- **State:** The current situation
  - position of a robot
- **Action:** The decision made by the agent
  - move left, move right
- **Reward:** The feedback from the environment
- **discount factor:** The importance of future rewards
- **return:** The sum of the rewards
- **policy:** The strategy used to select actions
  - position is the inupt, the output is the action to take

### State Action Value Function

The state action value function $Q(s, a)$ is the expected return starting from state $s$, taking action $a$ and following policy $\pi$.

At every state pick the action that maximizes the state action value function. 

The best possible return from state $s$ is the maximum of the state action value function over all actions.

```math
max_a Q(s, a)
```

Example:

$Q(s, a)$ = Return if you
- start in state $s$
- take action $a$ once
- then behave optimally after that


State 1 has a reward of 100 and state 6 has a reward of 40, all other states have a reward of 0. Starting at state 5, the best action is to move to the right. For states 2 - 4 the best action is to move to the left.

![alt text](images/state-action-function.png)

**Bellman Equation**

$Q(s, a)$ is the expected return form the reward of the current state $s$ and the max of all possible actions in the next state $s'$ multiplied by the discount factor $\gamma$.

```math
Q(s, a) = R(s) + \gamma * max_{a'} Q(s', a')
```

Expected return in stochastic environment:
```math
Q(s, a) = E(R_1 + \gamma R_2 + \gamma^2 R_3 + \ldots)
```
```math
Q(s, a) = R(s) + \gamma * E(max_{a'} Q(s', a'))
```

### Deep Reinforcement Learning

If both the state and action are discret values, we can estimate the action-value function iteratively. However, if the state and action space are continuous, we can use a neural network to estimate the action-value function.

The input to the neural network is the state $s$ and the action $a$. The ouput is the state action value function $Q(s, a)$ or $y$.

```math
\vec{x} = \begin{bmatrix} s \\ a \end{bmatrix} \Rightarrow \text{neuronal network} \Rightarrow  Q(s, a) = y
```

To train the network we need to create a training set and then train the network with supervised learning.
To create the training set we can use the Bellman equation.

Learing Algorithm:

- Initialize the neural network with randomly as guess of $Q(s, a)$
- Repeate:
  - Take actions in the environment. Get $(s, a, R(s), s')$
  - Store 10000 most recent tuples $(s, a, R(s), s')$ in replay memory
  - Train neural network:
    - Create training set of 10000 examples $x = (s, a)$ and $y = R(s) + \gamma * max_{a'} Q(s', a')$
    - Train $Q_{new}$ such that $Q_{new}(s, a) \approx y$
  - Set Q = $Q_{new}$

At the end the inital random guess of $Q(s, a)$ will be replaced by the optimal $Q(s, a)$.
If we repeat this, we and up with a better and better $Q(s, a)$.

**Soft update:**

Instead of setting $Q = Q_{new}$ we can use a soft update where we update $Q$ with a small fraction of $Q_{new}$. This way we can avoid oscillations.

```math
Q(s, a) = (1 - \tau) Q(s, a) + \tau Q_{new}(s, a)
```
- $\tau$ is the update rate e.g. 0.01

**Greedy policy:**

If we always pick the optimal action, we might miss some states and never explore some paths. To avoid this we can use the $\epsilon$-greedy policy that selects a random action with probability $\epsilon$ and the optimal action with probability $1 - \epsilon$.

```math
\pi(a|s) = \begin{cases}
1 - \epsilon + \frac{\epsilon}{|A|} & \text{if } a = \text{argmax}_{a'} Q(s, a') \\
\frac{\epsilon}{|A|} & \text{otherwise}
\end{cases}
```
- $|A|$ is the number of actions
- $Q(s, a)$ is the state action value function
- $a$ is the action
- $s$ is the state
- $\epsilon$ is the probability of selecting a random action
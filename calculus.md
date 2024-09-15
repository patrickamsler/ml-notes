# Calculus

- [Calculus](#calculus)
  - [Basic Rules](#basic-rules)
    - [First Derivative](#first-derivative)
    - [Second Derivative](#second-derivative)
    - [Chain Rule](#chain-rule)
  - [Calculating Partial Derivatives of a Linear Function and the Squared Error Cost Function](#calculating-partial-derivatives-of-a-linear-function-and-the-squared-error-cost-function)


## Basic Rules

### First Derivative

The first derivative of a function measures the rate of change or slope of the function at any given point.
If $f(x)$ is a function, the first derivative, $f'(x)$, tells us how $f(x)$ is changing with respect to $x$.

Interpretation:
- If $f'(x) > 0$, the function is increasing at that point.
- If $f'(x) < 0$, the function is decreasing at that point.
- If $f'(x) = 0$, the function has a horizontal tangent, which could indicate a local maximum, minimum, or a saddle point.

### Second Derivative

The second derivative of a function measures how the rate of change (first derivative) is itself changing. If  $f'(x)$  represents the slope,  $f''(x)$  represents how the slope is changing. In other words, it tells us about the curvature or concavity of the function

Interpretation:
- If $f''(x) > 0$, the function is concave up (like a smile) at that point.
- If $f''(x) < 0$, the function is concave down (like a frown) at that point.
- If $f''(x) = 0$, the function may have an inflection point, where concavity changes.

Critical Points:
- If $f'(x) = 0$ and $f''(x) > 0$, there is a local minimum at that point.
- If $f'(x) = 0$ and $f''(x) < 0$, there is a local maximum at that point.
- If $f'(x) = 0$ and $f''(x) = 0$, further investigation is required.

### Chain Rule

Leibniz's notation:

$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}
$$

Prime notation:

$$
(f(g(x)))' = f'(g(x)) \cdot g'(x)
$$

## Calculating Partial Derivatives of a Linear Function and the Squared Error Cost Function

Given a linear function $f_{wb}(x) = wx + b$ and the squared error cost function $J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{wb}(x^{(i)}) - y^{(i)} \right)^2$, we can calculate the partial derivatives with respect to $w$ and $b$.

Expand the cost function:

$$
J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( (wx^{(i)} + b) - y^{(i)} \right)^2
$$

Partial Derivative with Respect to $w$:

Apply the chain rule:

$$
\frac{\partial}{\partial w} \left( (wx^{(i)} + b) - y^{(i)} \right)^2 = 2 \left( (wx^{(i)} + b) - y^{(i)} \right) \cdot \frac{\partial}{\partial w} (wx^{(i)} + b - y^{(i)})
$$

$$
\frac{\partial}{\partial w} \left( (wx^{(i)} + b) - y^{(i)} \right)^2 = 2 \left( (wx^{(i)} + b) - y^{(i)} \right) x^{(i)}
$$

sum over all training samples (2 cancels out with 1/2m):

$$
\frac{\partial}{\partial w} J(w, b) = \frac{1}{m} \sum_{i=1}^{m} \left( wx^{(i)} + b - y^{(i)} \right) x^{(i)}
$$

Partial Derivative with Respect to $b$:

Apply the chain rule:

$$
\frac{\partial}{\partial b} \left( (wx^{(i)} + b) - y^{(i)} \right)^2 = 2 \left( (wx^{(i)} + b) - y^{(i)} \right) \cdot \frac{\partial}{\partial b} (wx^{(i)} + b - y^{(i)})
$$

$$
\frac{\partial}{\partial b} \left( (wx^{(i)} + b) - y^{(i)} \right)^2 = 2 \left( (wx^{(i)} + b) - y^{(i)} \right)
$$

sum over all training samples (2 cancels out with 1/2m):

$$
\frac{\partial}{\partial b} J(w, b) = \frac{1}{m} \sum_{i=1}^{m} \left( wx^{(i)} + b - y^{(i)} \right)
$$
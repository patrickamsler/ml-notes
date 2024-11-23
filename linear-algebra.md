# Linear Algebra

- [Linear Algebra](#linear-algebra)
  - [Dot Product](#dot-product)
  - [Square of a Vector](#square-of-a-vector)
  - [Matrix Transpose](#matrix-transpose)
  - [Matrix Multiplication](#matrix-multiplication)
  - [Euclidean Distance](#euclidean-distance)


## Dot Product
For two vectors with the same dimension **$\vec{w}$** and **$\vec{x}$**, each with $n$ entries, the dot product is calculated as:

```math
\vec{w} \cdot \vec{x} = w_1x_1 + w_2x_2 + \dots + w_nx_n
```

Let **$\vec{w}$** and **$\vec{x}$** be two vectors with 4 entries:

```math
\vec{w} = [w_1, w_2, w_3, w_4] = [2, -1, 0.5, 3]
```

```math
\vec{x} = [x_1, x_2, x_3, x_4] = [1, 0.5, -2, 4]
```

```math
\vec{w} \cdot \vec{x} = (2)(1) + (-1)(0.5) + (0.5)(-2) + (3)(4) = 12.5
```

## Square of a Vector
the square of the vector, $\vec{v}$,  (squared magnitude or the dot product of the vector with itself)

```math
\vec{v} \cdot \vec{v} = v_1^2 + v_2^2 + \dots + v_n^2
```

```math
\mathbf{v}^T \mathbf{v} = v_1^2 + v_2^2 + \dots v_n^2
```

Example with a vector $\vec{v} = [1, 2, 3]$:

```math
\vec{v} \cdot \vec{v} = (1)^2 + (2)^2 + (3)^2 = 14
```

## Matrix Transpose

The transpose of a matrix is obtained by swapping the rows and columns.

```math
\mathbf{A} =
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}
```

```math
\mathbf{A}^T =
\begin{bmatrix}
1 & 4 \\
2 & 5 \\
3 & 6
\end{bmatrix}
```

## Matrix Multiplication

Given matrices $\mathbf{A}$ of dimensions $m \times n$ and $\mathbf{B}$ of dimensions $n \times p$, the product $\mathbf{C}$ will have dimensions $m \times p$. The elements of $\mathbf{C}$ are computed as:

```math
C_{mp} = A_{mn} \times B_{np}
```

- Matrix multiplication is not commutative, meaning $A \times B \neq B \times A$ in general.
- It is associative, so $(A \times B) \times C = A \times (B \times C)$.
- If the number of columns $n$ in $\mathbf{A}$ doesn’t match the number of rows $n$ in $\mathbf{B}$, matrix multiplication is not possible

Matrix $\mathbf{A}$ (3 rows, 2 columns):

```math
\mathbf{A} =
\begin{bmatrix}
1 & 2 \\
3 & 4 \\
5 & 6
\end{bmatrix}
```

Matrix $\mathbf{B}$ (2 rows, 4 columns):

```math
\mathbf{B} =
\begin{bmatrix}
7 & 8 & 9 & 10 \\
11 & 12 & 13 & 14
\end{bmatrix}
```

The product $\mathbf{C} = \mathbf{A} \times \mathbf{B}$ will have dimensions 3 rows and 4 columns:

```math
\mathbf{C} =
\begin{bmatrix}
29 & 32 & 35 & 38 \\
65 & 72 & 79 & 86 \\
101 & 112 & 123 & 134
\end{bmatrix}
```

The elements of $\mathbf{C}$ are computed as follows:

```math
\mathbf{C}_{11} = (1 \cdot 7) + (2 \cdot 11) = 7 + 22 = 29
```

```math
\mathbf{C}_{12} = (1 \cdot 8) + (2 \cdot 12) = 8 + 24 = 32
```

...

```math
\mathbf{C}_{34} = (5 \cdot 10) + (6 \cdot 14) = 50 + 84 = 134
```

## Euclidean Distance
For two vectors $\mathbf{x} = (x_1, x_2, \dots, x_n)$ and $\mathbf{y} = (y_1, y_2, \dots, y_n)$, the Euclidean distance between them is calculated as:

```math
\|\mathbf{x} - \mathbf{y}\| = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}
```

Alternatively, it can be written as:

```math
\|\mathbf{x} - \mathbf{y}\|^2 = \sum_{i=1}^n (x_i - y_i)^2
```

- $\|\mathbf{x} - \mathbf{y}\|$: Denotes the L2 norm. Can also be written as $\|\mathbf{x} - \mathbf{y}\|_2$. If there is no number specified, it is assumed to be 2.
- $n$: The number of dimensions in the vectors $\mathbf{x}$ and $\mathbf{y}$.

**Example**:

Vectors:

$\mathbf{x} = (2, 3, 5)$, $\mathbf{y} = (6, 8, 10)$

Distance:

```math
\|\mathbf{x} - \mathbf{y}\|_2 = \sqrt{(2 - 6)^2 + (3 - 8)^2 + (5 - 10)^2}
```

```math
\|\mathbf{x} - \mathbf{y}\|_2 = \sqrt{(-4)^2 + (-5)^2 + (-5)^2} = \sqrt{16 + 25 + 25} = \sqrt{66} \approx 8.124
```

**Special Cases**:

**Norm of a Single Vector**:
The L2 norm of a single vector $\mathbf{x}$ is the distance of the vector from its origin:

```math
\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2}
```

**In Terms of Matrix Operations**:
If $\mathbf{x}$ and $\mathbf{y}$ are vectors, the Euclidean distance can also be written as:

```math
\|\mathbf{x} - \mathbf{y}\|_2 = \sqrt{(\mathbf{x} - \mathbf{y})^\top (\mathbf{x} - \mathbf{y})}
```
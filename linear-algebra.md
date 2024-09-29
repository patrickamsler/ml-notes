# Linear Algebra

- [Linear Algebra](#linear-algebra)
  - [Dot Product](#dot-product)
  - [Square of a Vector](#square-of-a-vector)
  - [Matrix Multiplication](#matrix-multiplication)


## Dot Product
For two vectors with the same dimension **$\vec{w}$** and **$\vec{x}$**, each with $n$ entries, the dot product is calculated as:

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

## Square of a Vector
the square of the vector, $\vec{v}$,  (squared magnitude or the dot product of the vector with itself)

$$
\vec{v} \cdot \vec{v} = v_1^2 + v_2^2 + \dots + v_n^2
$$

$$
\mathbf{v}^T \mathbf{v} = v_1^2 + v_2^2 + \dots v_n^2
$$

Example with a vector $\vec{v} = [1, 2, 3]$:

$$
\vec{v} \cdot \vec{v} = (1)^2 + (2)^2 + (3)^2 = 14
$$

## Matrix Multiplication

Given matrices $\mathbf{A}$ of dimensions $m \times n$ and $\mathbf{B}$ of dimensions $n \times p$, the product $\mathbf{C}$ will have dimensions $m \times p$. The elements of $\mathbf{C}$ are computed as:

$$
C_{mp} = A_{mn} \times B_{np}
$$

- Matrix multiplication is not commutative, meaning $A \times B \neq B \times A$ in general.
- It is associative, so $(A \times B) \times C = A \times (B \times C)$.
- If the number of columns $n$ in $\mathbf{A}$ doesn’t match the number of rows $n$ in $\mathbf{B}$, matrix multiplication is not possible

$$
\mathbf{A} =
\begin{pmatrix}
1 & 2 \\
3 & 4 \\
5 & 6
\end{pmatrix}
$$

Matrix $\mathbf{B}$ (2 rows, 4 columns):

$$
\mathbf{B} =
\begin{pmatrix}
7 & 8 & 9 & 10 \\
11 & 12 & 13 & 14
\end{pmatrix}
$$

The product $\mathbf{C} = \mathbf{A} \times \mathbf{B}$ will have dimensions 3 rows and 4 columns:

$$
\mathbf{C} =
\begin{pmatrix}
29 & 32 & 35 & 38 \\
65 & 72 & 79 & 86 \\
101 & 112 & 123 & 134
\end{pmatrix}
$$

The elements of $\mathbf{C}$ are computed as follows:

$$
\mathbf{C}_{11} = (1 \cdot 7) + (2 \cdot 11) = 7 + 22 = 29
$$

$$
\mathbf{C}_{12} = (1 \cdot 8) + (2 \cdot 12) = 8 + 24 = 32
$$

...

$$
\mathbf{C}_{34} = (5 \cdot 10) + (6 \cdot 14) = 50 + 84 = 134
$$
# Algebra — Complete Concept Reference for AI

A comprehensive guide to all core and advanced algebra concepts used in mathematics, machine learning, and AI systems.

---

## Table of Contents

1. [Foundations & Number Systems](#1-foundations--number-systems)
2. [Variables, Expressions & Equations](#2-variables-expressions--equations)
3. [Linear Equations & Inequalities](#3-linear-equations--inequalities)
4. [Systems of Equations](#4-systems-of-equations)
5. [Polynomials](#5-polynomials)
6. [Factoring](#6-factoring)
7. [Quadratic Equations](#7-quadratic-equations)
8. [Functions](#8-functions)
9. [Exponents & Logarithms](#9-exponents--logarithms)
10. [Rational Expressions](#10-rational-expressions)
11. [Radical Expressions & Complex Numbers](#11-radical-expressions--complex-numbers)
12. [Sequences & Series](#12-sequences--series)
13. [Matrices & Linear Algebra](#13-matrices--linear-algebra)
14. [Vectors](#14-vectors)
15. [Probability & Combinatorics](#15-probability--combinatorics)
16. [Algebra in AI & Machine Learning](#16-algebra-in-ai--machine-learning)
17. [Quick Reference Card](#17-quick-reference-card)

---

## 1. Foundations & Number Systems

### Number Sets

| Symbol | Set | Examples |
|--------|-----|---------|
| ℕ | Natural numbers | 1, 2, 3, 4, ... |
| ℤ | Integers | ..., −2, −1, 0, 1, 2, ... |
| ℚ | Rational numbers | 1/2, 0.75, −3/4 |
| ℝ | Real numbers | π, √2, 0.333..., 5 |
| ℂ | Complex numbers | 3 + 2i, −i, 4 |

### Properties of Real Numbers

| Property | Addition | Multiplication |
|----------|----------|----------------|
| Commutative | a + b = b + a | a × b = b × a |
| Associative | (a+b)+c = a+(b+c) | (ab)c = a(bc) |
| Distributive | a(b + c) = ab + ac | — |
| Identity | a + 0 = a | a × 1 = a |
| Inverse | a + (−a) = 0 | a × (1/a) = 1 |

### Order of Operations (PEMDAS)

```
P → Parentheses
E → Exponents
M → Multiplication
D → Division
A → Addition
S → Subtraction
```

---

## 2. Variables, Expressions & Equations

### Key Definitions

- **Variable** — a symbol (usually x, y) representing an unknown value
- **Constant** — a fixed number (e.g., 5, −3, π)
- **Coefficient** — the number multiplied by a variable (in `3x`, the coefficient is 3)
- **Term** — a single number, variable, or product (e.g., `5x²`, `−3y`, `7`)
- **Expression** — a combination of terms (e.g., `3x² + 2x − 5`)
- **Equation** — two expressions set equal (e.g., `3x + 2 = 11`)
- **Identity** — an equation true for all values (e.g., `(a+b)² = a²+2ab+b²`)

### Simplifying Expressions

**Combining like terms:**
```
3x² + 5x − 2x² + 4x − 7
= (3x² − 2x²) + (5x + 4x) − 7
= x² + 9x − 7
```

**Distributive property:**
```
4(3x − 2) + 5x
= 12x − 8 + 5x
= 17x − 8
```

---

## 3. Linear Equations & Inequalities

### Linear Equations

**Standard form:** `ax + b = c`

**Solution:**
$$x = \frac{c - b}{a}, \quad a \neq 0$$

**Worked example — Solve `5x − 3 = 22`:**

| Step | Operation | Result |
|------|-----------|--------|
| 1 | Add 3 to both sides | 5x = 25 |
| 2 | Divide both sides by 5 | **x = 5** |

---

### Two-Step Linear: `ax + b = cx + d`

$$x = \frac{d - b}{a - c}, \quad a \neq c$$

**Example — Solve `7x + 4 = 3x + 20`:**

| Step | Operation | Result |
|------|-----------|--------|
| 1 | Subtract 3x | 4x + 4 = 20 |
| 2 | Subtract 4 | 4x = 16 |
| 3 | Divide by 4 | **x = 4** |

---

### Linear Inequalities

| Inequality | Symbol | Graph |
|------------|--------|-------|
| Less than | x < a | open circle, arrow left |
| Less than or equal | x ≤ a | closed circle, arrow left |
| Greater than | x > a | open circle, arrow right |
| Greater than or equal | x ≥ a | closed circle, arrow right |

**Key rule:** When multiplying or dividing by a **negative** number, flip the inequality sign.

**Example — Solve `−2x + 6 > 2`:**
```
−2x > −4
x < 2    ← sign flipped (divided by −2)
```

---

### Absolute Value Equations

`|x| = a`  →  `x = a` or `x = −a`

`|x − 3| = 7`  →  `x = 10` or `x = −4`

---

## 4. Systems of Equations

### Two-Variable Systems

**Three solution types:**

| Type | Lines | Solutions |
|------|-------|-----------|
| Consistent & independent | Intersecting | One solution |
| Consistent & dependent | Same line | Infinite solutions |
| Inconsistent | Parallel | No solution |

---

### Method 1 — Substitution

**Example:** Solve `y = 2x − 1` and `x + y = 8`

```
Substitute y:  x + (2x − 1) = 8
               3x − 1 = 8
               3x = 9  →  x = 3
Back-substitute: y = 2(3) − 1 = 5
```
**Solution: (3, 5)**

---

### Method 2 — Elimination

**Example:** Solve `2x + 3y = 13` and `4x − 3y = 5`

```
Add equations:  6x = 18  →  x = 3
Substitute:     2(3) + 3y = 13  →  3y = 7  →  y = 7/3
```
**Solution: (3, 7/3)**

---

### Method 3 — Matrix (Cramer's Rule)

For `ax + by = e` and `cx + dy = f`:

$$x = \frac{ed - bf}{ad - bc}, \quad y = \frac{af - ec}{ad - bc}$$

The denominator `D = ad − bc` is the **determinant**.
If `D = 0`, no unique solution exists.

---

### Three-Variable Systems

**General form:**
```
a₁x + b₁y + c₁z = d₁
a₂x + b₂y + c₂z = d₂
a₃x + b₃y + c₃z = d₃
```
Solved using **Gaussian elimination** or **matrix inversion**.

---

## 5. Polynomials

### Definitions

- **Degree** — the highest exponent in the polynomial
- **Leading coefficient** — coefficient of the highest-degree term
- **Monomial** — 1 term (e.g., `5x³`)
- **Binomial** — 2 terms (e.g., `3x + 4`)
- **Trinomial** — 3 terms (e.g., `x² + 5x + 6`)

### Degree Classification

| Degree | Name | Example |
|--------|------|---------|
| 0 | Constant | 7 |
| 1 | Linear | 3x + 2 |
| 2 | Quadratic | x² − 4x + 4 |
| 3 | Cubic | 2x³ + x − 5 |
| 4 | Quartic | x⁴ − 3x² + 1 |
| 5 | Quintic | x⁵ − x |

### Polynomial Operations

**Addition:** combine like terms
```
(3x² + 2x − 1) + (x² − 5x + 4) = 4x² − 3x + 3
```

**Multiplication (FOIL for binomials):**
```
(x + 3)(x − 2)
= x² − 2x + 3x − 6
= x² + x − 6
```

**Special Products:**

| Pattern | Formula |
|---------|---------|
| Sum × Difference | (a+b)(a−b) = a² − b² |
| Perfect Square (sum) | (a+b)² = a² + 2ab + b² |
| Perfect Square (diff) | (a−b)² = a² − 2ab + b² |
| Cube sum | (a+b)³ = a³ + 3a²b + 3ab² + b³ |
| Cube difference | (a−b)³ = a³ − 3a²b + 3ab² − b³ |

### Polynomial Division

**Long division** and **synthetic division** are used to divide polynomials.

**Remainder Theorem:** If polynomial `P(x)` is divided by `(x − c)`, the remainder is `P(c)`.

**Factor Theorem:** `(x − c)` is a factor of `P(x)` if and only if `P(c) = 0`.

---

## 6. Factoring

### Greatest Common Factor (GCF)

```
6x³ + 9x² − 3x = 3x(2x² + 3x − 1)
```

### Factoring Trinomials `x² + bx + c`

Find `p` and `q` where `p + q = b` and `p × q = c`:
```
x² + 7x + 12 = (x + 3)(x + 4)   [3+4=7, 3×4=12]
```

### Factoring `ax² + bx + c` (AC Method)

1. Multiply `a × c`
2. Find two numbers that multiply to `ac` and add to `b`
3. Rewrite and factor by grouping

**Example:** Factor `2x² + 7x + 3`
```
a×c = 6,  find 6 and 1 (6+1=7, 6×1=6)
= 2x² + 6x + x + 3
= 2x(x + 3) + 1(x + 3)
= (2x + 1)(x + 3)
```

### Special Factoring Formulas

| Pattern | Formula |
|---------|---------|
| Difference of squares | a² − b² = (a+b)(a−b) |
| Sum of cubes | a³ + b³ = (a+b)(a²−ab+b²) |
| Difference of cubes | a³ − b³ = (a−b)(a²+ab+b²) |
| Perfect square trinomial | a²+2ab+b² = (a+b)² |

---

## 7. Quadratic Equations

### Three Solution Methods

#### Method 1 — Factoring
```
x² − 5x + 6 = 0
(x − 2)(x − 3) = 0
x = 2  or  x = 3
```

#### Method 2 — Completing the Square
```
x² + 6x + 5 = 0
x² + 6x = −5
x² + 6x + 9 = −5 + 9        ← add (b/2)² = 9
(x + 3)² = 4
x + 3 = ±2
x = −1  or  x = −5
```

#### Method 3 — Quadratic Formula

For `ax² + bx + c = 0`:

$$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$

**Discriminant analysis:**

| Δ = b² − 4ac | Nature of roots |
|---------------|----------------|
| Δ > 0 | Two distinct real roots |
| Δ = 0 | One repeated real root |
| Δ < 0 | Two complex conjugate roots |

**Example — Solve `2x² − 4x − 6 = 0`:**

```
a=2, b=−4, c=−6
Δ = 16 + 48 = 64
x = (4 ± 8) / 4
x = 3  or  x = −1
```

### Parabola Properties

| Property | Formula |
|----------|---------|
| Vertex | `(−b/2a, f(−b/2a))` |
| Axis of symmetry | `x = −b/2a` |
| y-intercept | `(0, c)` |
| x-intercepts (roots) | From quadratic formula |
| Opens up | `a > 0` |
| Opens down | `a < 0` |
| Wider than y=x² | `\|a\| < 1` |
| Narrower than y=x² | `\|a\| > 1` |

---

## 8. Functions

### Definition

A **function** `f` maps each input `x` to exactly one output `f(x)`.

`f : X → Y`  means f maps domain X to codomain Y.

### Function Notation

| Notation | Meaning |
|----------|---------|
| `f(x)` | Output of f at input x |
| `f(a)` | Evaluate f at x = a |
| `(f ∘ g)(x)` | f(g(x)) — composition |
| `f⁻¹(x)` | Inverse function |

### Types of Functions

| Type | Form | Graph |
|------|------|-------|
| Linear | f(x) = mx + b | Straight line |
| Quadratic | f(x) = ax² + bx + c | Parabola |
| Cubic | f(x) = ax³ + ... | S-curve |
| Absolute value | f(x) = \|x\| | V-shape |
| Square root | f(x) = √x | Half-parabola |
| Exponential | f(x) = aˣ | Exponential curve |
| Logarithmic | f(x) = log(x) | Slow growth curve |
| Rational | f(x) = p(x)/q(x) | With asymptotes |

### Domain & Range

- **Domain** — all valid input values (x)
- **Range** — all possible output values (y)

| Function | Domain | Range |
|----------|--------|-------|
| f(x) = x² | All reals | y ≥ 0 |
| f(x) = √x | x ≥ 0 | y ≥ 0 |
| f(x) = 1/x | x ≠ 0 | y ≠ 0 |
| f(x) = log(x) | x > 0 | All reals |

### Transformations

| Transformation | Rule | Effect |
|----------------|------|--------|
| Vertical shift up | f(x) + k | Moves graph up by k |
| Vertical shift down | f(x) − k | Moves graph down by k |
| Horizontal shift right | f(x − h) | Moves right by h |
| Horizontal shift left | f(x + h) | Moves left by h |
| Vertical stretch | a·f(x), a>1 | Stretches vertically |
| Vertical compress | a·f(x), 0<a<1 | Compresses vertically |
| Reflection (x-axis) | −f(x) | Flips over x-axis |
| Reflection (y-axis) | f(−x) | Flips over y-axis |

### Inverse Functions

If `f(a) = b`, then `f⁻¹(b) = a`.

**Steps to find inverse:**
1. Replace `f(x)` with `y`
2. Swap `x` and `y`
3. Solve for `y`
4. Replace `y` with `f⁻¹(x)`

**Example:** Find inverse of `f(x) = 3x − 7`
```
y = 3x − 7
x = 3y − 7
x + 7 = 3y
f⁻¹(x) = (x + 7) / 3
```

### Composition of Functions

`(f ∘ g)(x) = f(g(x))`

**Example:** `f(x) = x² + 1`, `g(x) = 2x`
```
(f ∘ g)(x) = f(2x) = (2x)² + 1 = 4x² + 1
(g ∘ f)(x) = g(x² + 1) = 2(x² + 1) = 2x² + 2
```

---

## 9. Exponents & Logarithms

### Laws of Exponents

| Law | Rule | Example |
|-----|------|---------|
| Product | xᵃ · xᵇ = xᵃ⁺ᵇ | x³ · x² = x⁵ |
| Quotient | xᵃ / xᵇ = xᵃ⁻ᵇ | x⁵ / x² = x³ |
| Power of power | (xᵃ)ᵇ = xᵃᵇ | (x²)³ = x⁶ |
| Power of product | (xy)ᵃ = xᵃyᵃ | (2x)³ = 8x³ |
| Zero exponent | x⁰ = 1 | 7⁰ = 1 |
| Negative exponent | x⁻ᵃ = 1/xᵃ | x⁻³ = 1/x³ |
| Fractional exponent | x^(1/n) = ⁿ√x | x^(1/2) = √x |
| General fractional | x^(m/n) = (ⁿ√x)ᵐ | x^(2/3) = (∛x)² |

### Exponential Functions

`f(x) = aˣ` where `a > 0, a ≠ 1`

| Property | Detail |
|----------|--------|
| Base a > 1 | Exponential growth |
| Base 0 < a < 1 | Exponential decay |
| y-intercept | Always (0, 1) |
| Domain | All real numbers |
| Range | y > 0 |
| Euler's number e | e ≈ 2.71828 |

### Logarithms

**Definition:** `logₐ(x) = y` means `aʸ = x`

$$\log_a(x) = y \iff a^y = x$$

| Log form | Exponential form |
|----------|-----------------|
| log₂(8) = 3 | 2³ = 8 |
| log₁₀(100) = 2 | 10² = 100 |
| ln(e²) = 2 | e² = e² |

### Logarithm Laws

| Law | Formula |
|-----|---------|
| Product | logₐ(xy) = logₐ(x) + logₐ(y) |
| Quotient | logₐ(x/y) = logₐ(x) − logₐ(y) |
| Power | logₐ(xⁿ) = n · logₐ(x) |
| Change of base | logₐ(x) = log(x) / log(a) |
| Identity | logₐ(a) = 1 |
| Zero | logₐ(1) = 0 |

### Natural Log (ln)

- `ln(x) = logₑ(x)` where `e ≈ 2.71828`
- `ln(eˣ) = x`
- `e^(ln x) = x`

**Solving exponential equations:**
```
2^x = 32
log₂(2^x) = log₂(32)
x = 5
```

**Solving log equations:**
```
log₃(x) = 4
x = 3⁴ = 81
```

---

## 10. Rational Expressions

### Definition

A rational expression is a fraction where numerator and/or denominator are polynomials: `P(x) / Q(x)`

### Simplifying

Factor completely, then cancel common factors:
```
(x² − 9) / (x + 3) = (x+3)(x−3) / (x+3) = x − 3   [x ≠ −3]
```

### Operations

| Operation | Rule |
|-----------|------|
| Multiply | (a/b)(c/d) = ac/bd |
| Divide | (a/b) ÷ (c/d) = ad/bc |
| Add/Subtract | Find LCD, then combine numerators |

### Partial Fractions

Decompose `(5x + 1) / ((x+1)(x−2))` into:
```
A/(x+1) + B/(x−2)
5x + 1 = A(x−2) + B(x+1)
→ A = 1, B = 4
= 1/(x+1) + 4/(x−2)
```

---

## 11. Radical Expressions & Complex Numbers

### Radicals

`ⁿ√x = x^(1/n)`

**Simplifying:**
```
√48 = √(16 × 3) = 4√3
```

**Operations:**
```
√3 × √12 = √36 = 6
(2√5)² = 4 × 5 = 20
```

**Rationalizing denominators:**
```
1/√2 = √2/2
1/(3 + √2) = (3 − √2) / (9 − 2) = (3 − √2)/7
```

### Complex Numbers

**Definition:** `i = √(−1)`, so `i² = −1`

**Form:** `a + bi` where `a` = real part, `b` = imaginary part

| Powers of i | Value |
|-------------|-------|
| i⁰ | 1 |
| i¹ | i |
| i² | −1 |
| i³ | −i |
| i⁴ | 1 (cycle repeats) |

**Operations:**
```
(3 + 2i) + (1 − 4i) = 4 − 2i
(3 + 2i)(1 − i) = 3 − 3i + 2i − 2i² = 3 − i + 2 = 5 − i
```

**Complex conjugate:** `a + bi` → conjugate is `a − bi`

**Modulus:** `|a + bi| = √(a² + b²)`

---

## 12. Sequences & Series

### Arithmetic Sequences

**Pattern:** each term increases by constant `d` (common difference)

**nth term:** `aₙ = a₁ + (n − 1)d`

**Sum of n terms:**
$$S_n = \frac{n}{2}(a_1 + a_n) = \frac{n}{2}(2a_1 + (n-1)d)$$

**Example:** 2, 5, 8, 11, ... (d = 3)
- a₁₀ = 2 + 9(3) = **29**
- S₁₀ = 10/2 × (2 + 29) = **155**

### Geometric Sequences

**Pattern:** each term multiplied by constant `r` (common ratio)

**nth term:** `aₙ = a₁ · rⁿ⁻¹`

**Sum of n terms:**
$$S_n = a_1 \cdot \frac{1 - r^n}{1 - r}, \quad r \neq 1$$

**Infinite geometric series** (|r| < 1):
$$S_\infty = \frac{a_1}{1 - r}$$

**Example:** 3, 6, 12, 24, ... (r = 2)
- a₅ = 3 · 2⁴ = **48**
- S₅ = 3(1 − 2⁵)/(1 − 2) = **93**

### Sigma Notation

$$\sum_{k=1}^{n} k = \frac{n(n+1)}{2}$$

$$\sum_{k=1}^{n} k^2 = \frac{n(n+1)(2n+1)}{6}$$

$$\sum_{k=1}^{n} k^3 = \left[\frac{n(n+1)}{2}\right]^2$$

### Binomial Theorem

$$(a + b)^n = \sum_{k=0}^{n} \binom{n}{k} a^{n-k} b^k$$

where `C(n,k) = n! / (k!(n−k)!)`

**Pascal's Triangle rows:**

```
n=0:     1
n=1:    1 1
n=2:   1 2 1
n=3:  1 3 3 1
n=4: 1 4 6 4 1
```

---

## 13. Matrices & Linear Algebra

### Matrix Basics

A matrix is a rectangular array of numbers arranged in rows and columns.

**Notation:** `A` is an `m × n` matrix (m rows, n columns)

```
A = | 1  2  3 |   (2×3 matrix)
    | 4  5  6 |
```

### Matrix Operations

**Addition/Subtraction** (same dimensions required):
```
|1 2| + |5 6| = |6  8 |
|3 4|   |7 8|   |10 12|
```

**Scalar multiplication:**
```
3 × |1 2| = |3  6|
    |3 4|   |9 12|
```

**Matrix multiplication** (A is m×n, B is n×p → result is m×p):

Element `C[i][j]` = dot product of row i of A and column j of B.

### Special Matrices

| Matrix | Description |
|--------|-------------|
| Square | n × n |
| Identity (I) | 1s on diagonal, 0s elsewhere |
| Zero | All elements are 0 |
| Diagonal | Non-zero only on diagonal |
| Symmetric | A = Aᵀ |
| Orthogonal | AᵀA = I |

### Determinant

For 2×2:
$$\det\begin{pmatrix}a & b \\ c & d\end{pmatrix} = ad - bc$$

For 3×3 (cofactor expansion):
$$\det(A) = a(ei-fh) - b(di-fg) + c(dh-eg)$$

### Matrix Inverse

`A⁻¹` exists only when `det(A) ≠ 0`.

For 2×2:
$$A^{-1} = \frac{1}{ad-bc}\begin{pmatrix}d & -b \\ -c & a\end{pmatrix}$$

**Property:** `A · A⁻¹ = A⁻¹ · A = I`

### Eigenvalues & Eigenvectors

For matrix `A`, eigenvector `v` and eigenvalue `λ` satisfy:

$$Av = \lambda v$$

**To find eigenvalues:** solve `det(A − λI) = 0` (characteristic equation)

**Significance in AI:**
- Principal Component Analysis (PCA) uses eigenvectors
- Dimensionality reduction relies on eigendecomposition
- PageRank algorithm is an eigenvector problem

### Gaussian Elimination

Used to solve systems of linear equations via row operations:

1. **Row swap:** R₁ ↔ R₂
2. **Row scale:** Rᵢ → k·Rᵢ
3. **Row addition:** Rᵢ → Rᵢ + k·Rⱼ

Goal: Reduce augmented matrix to **Row Echelon Form**:
```
|1 2  3 | 9  |         |1 0 0 | 1 |
|0 1  2 | 5  |  →  →  |0 1 0 | 2 |
|0 0  1 | 2  |         |0 0 1 | 2 |
```

---

## 14. Vectors

### Definition

A vector has both **magnitude** and **direction**.

`v = (v₁, v₂)` in 2D, or `v = (v₁, v₂, v₃)` in 3D

### Vector Operations

| Operation | Formula |
|-----------|---------|
| Addition | u + v = (u₁+v₁, u₂+v₂) |
| Scalar mult. | k·v = (kv₁, kv₂) |
| Magnitude | \|v\| = √(v₁² + v₂²) |
| Unit vector | v̂ = v / \|v\| |

### Dot Product

$$\mathbf{u} \cdot \mathbf{v} = u_1 v_1 + u_2 v_2 + \ldots = |\mathbf{u}||\mathbf{v}|\cos\theta$$

- If `u · v = 0`, vectors are **perpendicular**
- If `u · v > 0`, angle is acute
- If `u · v < 0`, angle is obtuse

### Cross Product (3D)

$$\mathbf{u} \times \mathbf{v} = (u_2v_3 - u_3v_2,\ u_3v_1 - u_1v_3,\ u_1v_2 - u_2v_1)$$

Result is a vector **perpendicular** to both u and v.

### Vector Projection

$$\text{proj}_{\mathbf{b}}\mathbf{a} = \frac{\mathbf{a} \cdot \mathbf{b}}{|\mathbf{b}|^2} \mathbf{b}$$

**AI relevance:** Vectors represent features, embeddings, word tokens, and data points in machine learning.

---

## 15. Probability & Combinatorics

### Counting Principles

**Multiplication rule:** If event A has m ways and event B has n ways, together they have `m × n` ways.

### Permutations (order matters)

$$P(n, r) = \frac{n!}{(n-r)!}$$

Arrange 3 from 5: P(5,3) = 60

### Combinations (order doesn't matter)

$$C(n, r) = \binom{n}{r} = \frac{n!}{r!(n-r)!}$$

Choose 3 from 5: C(5,3) = 10

### Basic Probability

$$P(A) = \frac{\text{favorable outcomes}}{\text{total outcomes}}$$

| Rule | Formula |
|------|---------|
| Complement | P(A') = 1 − P(A) |
| Union | P(A∪B) = P(A) + P(B) − P(A∩B) |
| Intersection (independent) | P(A∩B) = P(A)·P(B) |
| Conditional | P(A\|B) = P(A∩B) / P(B) |
| Bayes' Theorem | P(A\|B) = P(B\|A)·P(A) / P(B) |

---

## 16. Algebra in AI & Machine Learning

### Linear Regression

Fits a line `y = mx + b` (or hyperplane in higher dimensions) to minimize error.

**Cost function (MSE):**
$$J(\theta) = \frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)}) - y^{(i)})^2$$

**Normal equation (exact solution):**
$$\theta = (X^T X)^{-1} X^T y$$

---

### Gradient Descent

Iteratively update parameters to minimize a cost function:

$$\theta_{new} = \theta_{old} - \alpha \nabla J(\theta)$$

where `α` is the **learning rate**.

| Variant | Update frequency |
|---------|-----------------|
| Batch GD | Entire dataset |
| Stochastic GD | One sample |
| Mini-batch GD | Small batch |

---

### Activation Functions

| Function | Formula | Range |
|----------|---------|-------|
| Sigmoid | σ(x) = 1/(1+e⁻ˣ) | (0, 1) |
| Tanh | tanh(x) = (eˣ−e⁻ˣ)/(eˣ+e⁻ˣ) | (−1, 1) |
| ReLU | max(0, x) | [0, ∞) |
| Leaky ReLU | max(αx, x) | (−∞, ∞) |
| Softmax | eˣⁱ / Σeˣʲ | (0, 1), sum=1 |

---

### Loss Functions

| Task | Loss Function | Formula |
|------|--------------|---------|
| Regression | Mean Squared Error | (1/n)Σ(yᵢ − ŷᵢ)² |
| Regression | Mean Absolute Error | (1/n)Σ\|yᵢ − ŷᵢ\| |
| Binary classification | Binary Cross-Entropy | −Σ[y log(ŷ) + (1−y)log(1−ŷ)] |
| Multi-class | Categorical Cross-Entropy | −Σ yᵢ log(ŷᵢ) |

---

### Softmax & Probability

Converts raw scores (logits) to probabilities:

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$

---

### Dot Product in Neural Networks

Each neuron computes a weighted sum:

$$z = w_1x_1 + w_2x_2 + \ldots + w_nx_n + b = \mathbf{w} \cdot \mathbf{x} + b$$

then applies activation: `a = f(z)`

---

### Dimensionality & PCA

**Principal Component Analysis** uses eigenvectors to reduce dimensions while preserving variance:

1. Compute the covariance matrix `Σ`
2. Find eigenvectors and eigenvalues of `Σ`
3. Select top `k` eigenvectors (principal components)
4. Project data: `Z = XW`

---

### Transformer Attention (Self-Attention)

The core algebraic operation in modern LLMs:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where `Q`, `K`, `V` are matrices for Queries, Keys, and Values.

---

### Backpropagation (Chain Rule)

For a composition of functions `L = f(g(h(x)))`:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial f} \cdot \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial h} \cdot \frac{\partial h}{\partial x}$$

---

### Regularization

**L1 (Lasso):** adds `λΣ|wᵢ|` to loss → promotes sparsity

**L2 (Ridge):** adds `λΣwᵢ²` to loss → penalizes large weights

$$J_{reg}(\theta) = J(\theta) + \lambda \sum_j \theta_j^2$$

---

### Sigmoid & Logistic Regression

$$\hat{y} = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$$

Decision boundary: predict class 1 if `ŷ ≥ 0.5`, else class 0.

---

## 17. Quick Reference Card

### Core Formulas

| Concept | Formula |
|---------|---------|
| Linear equation | x = (c − b) / a |
| Quadratic formula | x = (−b ± √(b²−4ac)) / 2a |
| Discriminant | Δ = b² − 4ac |
| Vertex of parabola | x = −b/2a |
| Difference of squares | a²−b² = (a+b)(a−b) |
| Perfect square | (a±b)² = a²±2ab+b² |
| Sum/diff of cubes | a³±b³ = (a±b)(a²∓ab+b²) |
| Exponent product | xᵃ·xᵇ = xᵃ⁺ᵇ |
| Log product | log(xy) = log(x)+log(y) |
| Arithmetic nth term | aₙ = a₁ + (n−1)d |
| Geometric nth term | aₙ = a₁ · rⁿ⁻¹ |
| Infinite geometric sum | S = a₁/(1−r), \|r\|<1 |
| 2×2 Determinant | ad − bc |
| Dot product | u·v = \|u\|\|v\|cosθ |
| Combinations | C(n,r) = n!/r!(n−r)! |
| Permutations | P(n,r) = n!/(n−r)! |
| Bayes' Theorem | P(A\|B) = P(B\|A)P(A)/P(B) |
| Gradient descent | θ := θ − α∇J(θ) |
| Self-Attention | softmax(QKᵀ/√dₖ)V |

---

### Algebraic Identities Cheat Sheet

```
(a + b)²     = a² + 2ab + b²
(a − b)²     = a² − 2ab + b²
(a + b)(a−b) = a² − b²
(a + b)³     = a³ + 3a²b + 3ab² + b³
(a − b)³     = a³ − 3a²b + 3ab² − b³
a³ + b³      = (a + b)(a² − ab + b²)
a³ − b³      = (a − b)(a² + ab + b²)
```

---

*Generated with Claude · Comprehensive Algebra Reference for AI*

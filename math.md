# Math
1. Algebra
# Algebra Simulation — Maths Reference Guide

A structured reference covering core algebra topics with worked examples, formulas, and step-by-step solutions.

---

## 1. Linear Equations

### Form: `ax + b = c`

**Formula:**
$$x = \frac{c - b}{a}$$

**Steps to solve:**
1. Subtract `b` from both sides: `ax = c − b`
2. Divide both sides by `a`: `x = (c − b) / a`

**Example:** Solve `3x + 5 = 20`

| Step | Operation | Result |
|------|-----------|--------|
| Start | 3x + 5 = 20 | — |
| Subtract 5 | 3x = 20 − 5 | 3x = 15 |
| Divide by 3 | x = 15 ÷ 3 | **x = 5** |

---

### Two-Step Linear: `ax + b = cx + d`

**Steps to solve:**
1. Move all `x` terms to the left: `ax − cx = d − b`
2. Simplify: `(a − c)x = d − b`
3. Divide: `x = (d − b) / (a − c)`

**Example:** Solve `5x + 3 = 2x + 13`

| Step | Operation | Result |
|------|-----------|--------|
| Move x terms | 5x − 2x = 13 − 3 | 3x = 10 |
| Divide by 3 | x = 10 ÷ 3 | **x = 3.33** |

---

## 2. Quadratic Equations

### Form: `ax² + bx + c = 0`

**The Quadratic Formula:**
$$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$

**The Discriminant:** `Δ = b² − 4ac`

| Discriminant | Meaning |
|--------------|---------|
| `Δ > 0` | Two distinct real roots |
| `Δ = 0` | One repeated real root |
| `Δ < 0` | No real roots (complex roots) |

**Example:** Solve `x² − 5x + 6 = 0`

| Step | Working |
|------|---------|
| Identify | a = 1, b = −5, c = 6 |
| Discriminant | Δ = (−5)² − 4(1)(6) = 25 − 24 = **1** |
| √Δ | √1 = 1 |
| x₁ | (5 + 1) / 2 = **3** |
| x₂ | (5 − 1) / 2 = **2** |

**Roots: x = 3 and x = 2**

---

## 3. Graph of a Quadratic (Parabola)

### Form: `y = ax² + bx + c`

| Property | Formula |
|----------|---------|
| Vertex x-coordinate | `x = −b / (2a)` |
| Vertex y-coordinate | `y = c − b² / (4a)` |
| Axis of symmetry | `x = −b / (2a)` |
| Opens upward | when `a > 0` |
| Opens downward | when `a < 0` |
| y-intercept | `(0, c)` |

**Key shapes by `a`:**

```
a > 0 (opens up)       a < 0 (opens down)
      |                       *   *
      |  * *             *         *
      | *   *
      *     *
```

**Example:** `y = x² − 4x + 3`

- Vertex x = −(−4) / 2(1) = **2**
- Vertex y = 4 − 4 + 3 = **−1** → Vertex = **(2, −1)**
- Opens **upward** (a = 1 > 0)
- y-intercept = **(0, 3)**

---

## 4. Factoring

### Factoring `x² + bx + c`

Find two numbers `p` and `q` such that:
- `p + q = b`
- `p × q = c`

Then: `x² + bx + c = (x + p)(x + q)`

**Example:** Factor `x² + 5x + 6`

| Pair | Sum | Product |
|------|-----|---------|
| 2 and 3 | 5 ✓ | 6 ✓ |

**Result: (x + 2)(x + 3)**

---

### Difference of Squares

$$a^2 - b^2 = (a + b)(a - b)$$

**Examples:**

| Expression | Factored Form |
|------------|---------------|
| x² − 9 | (x + 3)(x − 3) |
| 25 − x² | (5 + x)(5 − x) |
| 4x² − 49 | (2x + 7)(2x − 7) |

---

### Perfect Square Trinomials

$$a^2 + 2ab + b^2 = (a + b)^2$$
$$a^2 - 2ab + b^2 = (a - b)^2$$

**Examples:**

| Expression | Factored Form |
|------------|---------------|
| x² + 6x + 9 | (x + 3)² |
| x² − 10x + 25 | (x − 5)² |

---

## 5. Laws of Exponents

| Law | Rule | Example |
|-----|------|---------|
| Product | `xᵃ × xᵇ = xᵃ⁺ᵇ` | x² × x³ = x⁵ |
| Quotient | `xᵃ ÷ xᵇ = xᵃ⁻ᵇ` | x⁵ ÷ x² = x³ |
| Power | `(xᵃ)ᵇ = xᵃᵇ` | (x²)³ = x⁶ |
| Zero | `x⁰ = 1` | 5⁰ = 1 |
| Negative | `x⁻ᵃ = 1/xᵃ` | x⁻² = 1/x² |

---

## 6. Systems of Linear Equations

### Substitution Method

**Example:** Solve `y = 2x + 1` and `3x + y = 16`

| Step | Working |
|------|---------|
| Substitute y | 3x + (2x + 1) = 16 |
| Simplify | 5x + 1 = 16 |
| Solve x | x = 3 |
| Solve y | y = 2(3) + 1 = **7** |

**Solution: x = 3, y = 7**

---

### Elimination Method

**Example:** Solve `2x + 3y = 12` and `4x − 3y = 6`

| Step | Working |
|------|---------|
| Add equations | 6x = 18 |
| Solve x | x = 3 |
| Substitute | 2(3) + 3y = 12 → 3y = 6 |
| Solve y | y = 2 |

**Solution: x = 3, y = 2**

---

## 7. Quick Reference Card

| Concept | Formula |
|---------|---------|
| Linear | x = (c − b) / a |
| Quadratic formula | x = (−b ± √(b²−4ac)) / 2a |
| Discriminant | Δ = b² − 4ac |
| Vertex of parabola | (−b/2a, c − b²/4a) |
| Difference of squares | a²−b² = (a+b)(a−b) |
| Perfect square | a²±2ab+b² = (a±b)² |
| Exponent product | xᵃ·xᵇ = xᵃ⁺ᵇ |

---

*Generated with Claude · Algebra Simulation*

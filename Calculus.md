# Calculus for AI — Complete Notes

> **Goal:** Build a solid, AI-ready understanding of calculus — from derivatives to backpropagation, optimization, and beyond.

---

## Table of Contents

1. [Why Calculus Matters in AI](#1-why-calculus-matters-in-ai)
2. [Limits & Continuity](#2-limits--continuity)
3. [Derivatives — Single Variable](#3-derivatives--single-variable)
4. [Differentiation Rules](#4-differentiation-rules)
5. [Partial Derivatives](#5-partial-derivatives)
6. [Gradients & Directional Derivatives](#6-gradients--directional-derivatives)
7. [The Chain Rule (Heart of Backprop)](#7-the-chain-rule-heart-of-backprop)
8. [Higher-Order Derivatives & Hessian](#8-higher-order-derivatives--hessian)
9. [Jacobian Matrix](#9-jacobian-matrix)
10. [Taylor Series & Approximation](#10-taylor-series--approximation)
11. [Multivariable Optimization](#11-multivariable-optimization)
12. [Gradient Descent & Variants](#12-gradient-descent--variants)
13. [Integrals — Concepts for AI](#13-integrals--concepts-for-ai)
14. [Calculus in Neural Networks (End-to-End)](#14-calculus-in-neural-networks-end-to-end)
15. [Activation Functions & Their Derivatives](#15-activation-functions--their-derivatives)
16. [Automatic Differentiation](#16-automatic-differentiation)
17. [Quick Reference Cheat Sheet](#17-quick-reference-cheat-sheet)

---

## 1. Why Calculus Matters in AI

Calculus is the engine behind **learning**. When a neural network trains, it is performing calculus millions of times per second.

| AI Task | Calculus Concept Used |
|---|---|
| Training neural networks | Gradient descent (derivatives) |
| Backpropagation | Chain rule |
| Loss minimization | Finding critical points |
| Optimization algorithms (Adam, RMSProp) | First & second-order derivatives |
| Regularization analysis | Taylor approximations |
| Attention mechanism gradients | Jacobian matrices |
| Normalizing flows (generative AI) | Change of variables (Jacobian determinant) |
| Physics-informed neural networks | Differential equations |

**Core idea:** To improve a model, you need to know *how much each parameter contributed to the error* — that is exactly what derivatives compute.

---

## 2. Limits & Continuity

### Limit
The value a function approaches as input approaches a point:
```
lim_{x → a} f(x) = L
```

**Intuition:** As x gets closer and closer to a, f(x) gets closer and closer to L.

### One-sided Limits
```
lim_{x → a⁻} f(x)   (from the left)
lim_{x → a⁺} f(x)   (from the right)
```
A limit exists iff both one-sided limits are equal.

### Key Limit Rules
```
lim [f(x) + g(x)] = lim f(x) + lim g(x)
lim [f(x) · g(x)] = lim f(x) · lim g(x)
lim [f(x) / g(x)] = lim f(x) / lim g(x)   (if denominator ≠ 0)
```

### L'Hôpital's Rule
For 0/0 or ∞/∞ indeterminate forms:
```
lim_{x→a} f(x)/g(x) = lim_{x→a} f'(x)/g'(x)
```

**AI use:** Analyzing behavior of activation functions at extremes (e.g., sigmoid as x → ±∞).

### Continuity
A function f is continuous at x = a if:
1. f(a) is defined
2. lim_{x→a} f(x) exists
3. lim_{x→a} f(x) = f(a)

**AI use:** Neural network activation functions must be continuous (and ideally differentiable) for gradient-based training to work.

---

## 3. Derivatives — Single Variable

### Definition
The derivative measures the **instantaneous rate of change** of a function:
```
f'(x) = df/dx = lim_{h→0} [f(x+h) − f(x)] / h
```

**Geometric interpretation:** Slope of the tangent line to f(x) at point x.

**AI interpretation:** How much does the output (loss) change if we slightly increase input (weight) x?

### Notation
All equivalent:
```
f'(x) = df/dx = d/dx[f(x)] = Df(x) = ẋ (time derivative)
```

### Differentiability
- A function is differentiable at x if the derivative exists at x
- Differentiable ⟹ Continuous (but not vice versa)
- **Non-differentiable points in AI:** ReLU at x = 0 (handled with subgradients)

### Common Derivatives
```
d/dx [c]      = 0             (constant)
d/dx [x]      = 1
d/dx [xⁿ]    = n·xⁿ⁻¹        (power rule)
d/dx [eˣ]    = eˣ
d/dx [aˣ]    = aˣ ln(a)
d/dx [ln x]  = 1/x
d/dx [log_a x] = 1/(x ln a)
d/dx [sin x] = cos x
d/dx [cos x] = −sin x
d/dx [tan x] = sec²x
d/dx [σ(x)]  = σ(x)(1 − σ(x))    (sigmoid — critical for backprop)
```

---

## 4. Differentiation Rules

### Sum / Difference Rule
```
d/dx [f(x) ± g(x)] = f'(x) ± g'(x)
```

### Product Rule
```
d/dx [f(x) · g(x)] = f'(x)·g(x) + f(x)·g'(x)
```

### Quotient Rule
```
d/dx [f(x)/g(x)] = [f'(x)·g(x) − f(x)·g'(x)] / [g(x)]²
```

### Power Rule
```
d/dx [xⁿ] = n · xⁿ⁻¹
```

### Chain Rule (Single Variable)
```
d/dx [f(g(x))] = f'(g(x)) · g'(x)
```

**Example:**
```
d/dx [sin(x²)] = cos(x²) · 2x
```

### Exponential & Log Rules
```
d/dx [eˣ]     = eˣ
d/dx [e^f(x)] = e^f(x) · f'(x)
d/dx [ln f(x)] = f'(x) / f(x)
```

**Log derivative trick** (used in policy gradient RL):
```
∇_θ log p(x;θ) = ∇_θ p(x;θ) / p(x;θ)
```

---

## 5. Partial Derivatives

When a function has **multiple inputs**, a partial derivative measures the rate of change with respect to one variable while holding all others constant.

### Notation
```
∂f/∂x    "partial f with respect to x"
```

### Example
```
f(x, y) = 3x²y + sin(y)

∂f/∂x = 6xy        (treat y as constant)
∂f/∂y = 3x² + cos(y)   (treat x as constant)
```

### Geometric Interpretation
The partial derivative ∂f/∂x is the slope of f in the x-direction (slice through the surface parallel to the x-axis).

### Mixed Partial Derivatives
```
∂²f/∂x∂y = ∂/∂x (∂f/∂y)
```

**Clairaut's theorem:** For smooth functions, mixed partials are equal:
```
∂²f/∂x∂y = ∂²f/∂y∂x
```

### AI Context
In a neural network with millions of weights w₁, w₂, ..., wₙ, training requires computing:
```
∂L/∂wᵢ  for every weight wᵢ
```
This is exactly computing partial derivatives of the loss L with respect to each weight.

---

## 6. Gradients & Directional Derivatives

### Gradient
The gradient is the **vector of all partial derivatives**. It points in the direction of steepest increase.

```
∇f(x) = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ
```

**Example:**
```
f(x, y) = x² + y²
∇f = [2x, 2y]ᵀ
```

### Properties of the Gradient
```
∇(f + g) = ∇f + ∇g
∇(cf)    = c·∇f
∇(fg)    = f·∇g + g·∇f
```

- ∇f points toward the **steepest ascent**
- −∇f points toward the **steepest descent** ← used in gradient descent
- |∇f| = magnitude of steepest slope

### Directional Derivative
Rate of change of f in direction unit vector u:
```
D_u f(x) = ∇f(x) · u = |∇f| cos(θ)
```
- Maximum when u aligns with ∇f (steepest ascent)
- Minimum when u opposes ∇f (steepest descent)
- Zero when u is perpendicular to ∇f

### Gradient in AI
The gradient of the loss function with respect to all weights is the most computed quantity in all of deep learning:
```
∇_W L(W) = [∂L/∂w₁, ∂L/∂w₂, ..., ∂L/∂wₙ]ᵀ
```

---

## 7. The Chain Rule (Heart of Backprop)

The chain rule is the **most important rule in AI calculus**. Backpropagation is nothing more than the chain rule applied repeatedly through a computational graph.

### Single Variable Chain Rule
```
If y = f(u) and u = g(x), then:
dy/dx = (dy/du) · (du/dx)
```

### Multi-Variable Chain Rule
```
If z = f(x, y) where x = g(t), y = h(t):
dz/dt = (∂z/∂x)(dx/dt) + (∂z/∂y)(dy/dt)
```

### Chain Rule for Vectors (General Form)
```
d/dx [f(g(x))] = [∂f/∂g] · [∂g/∂x]   (matrix multiplication of Jacobians)
```

### Chain Rule Through a Neural Network
Consider a 3-layer network:
```
Input x → Layer 1 (z₁ = W₁x + b₁) → a₁ = σ(z₁)
         → Layer 2 (z₂ = W₂a₁ + b₂) → a₂ = σ(z₂)
         → Loss L = ℓ(a₂, y)

∂L/∂W₁ = (∂L/∂a₂) · (∂a₂/∂z₂) · (∂z₂/∂a₁) · (∂a₁/∂z₁) · (∂z₁/∂W₁)
```

Each term is a local derivative; the chain rule multiplies them together.

### Why Backpropagation is Efficient
Naive computation would recompute the same sub-expressions many times. Backprop uses **dynamic programming** — computing and caching each local gradient once, then reusing:
- **Forward pass:** Compute and store intermediate activations
- **Backward pass:** Apply chain rule right-to-left, reusing stored values

```
Time complexity: O(forward pass) — same order as a single prediction
```

---

## 8. Higher-Order Derivatives & Hessian

### Second Derivative
```
f''(x) = d²f/dx² = d/dx[f'(x)]
```

**Meaning:**
- f'' > 0: f is **convex** (concave up) — local min possible
- f'' < 0: f is **concave** (concave down) — local max
- f'' = 0: inflection point (possibly)

### Hessian Matrix
For a function f : ℝⁿ → ℝ, the Hessian is the matrix of all second-order partial derivatives:

```
H(f) = ∇²f = 
    [∂²f/∂x₁²      ∂²f/∂x₁∂x₂  ···  ∂²f/∂x₁∂xₙ]
    [∂²f/∂x₂∂x₁    ∂²f/∂x₂²    ···  ∂²f/∂x₂∂xₙ]
    [      ⋮               ⋮       ⋱        ⋮     ]
    [∂²f/∂xₙ∂x₁   ∂²f/∂xₙ∂x₂  ···  ∂²f/∂xₙ²   ]
```

The Hessian is **symmetric** for smooth functions (Clairaut's theorem).

### Hessian & Optimization

| Hessian Property | Interpretation |
|---|---|
| Positive definite (all eigenvalues > 0) | Strict local minimum |
| Negative definite (all eigenvalues < 0) | Strict local maximum |
| Indefinite (mixed eigenvalues) | Saddle point |
| Singular (zero eigenvalue) | Degenerate — need higher-order analysis |

### AI Uses of the Hessian
- **Newton's method:** Uses H⁻¹∇f for faster convergence
- **Second-order optimization:** L-BFGS approximates the Hessian
- **Loss landscape analysis:** Flat minima (small Hessian eigenvalues) often generalize better
- **Computing: expensive** — Hessian for n parameters requires O(n²) memory, impractical for millions of weights → gradient-only methods dominate

---

## 9. Jacobian Matrix

When a function maps vectors to vectors f : ℝⁿ → ℝᵐ, the Jacobian generalizes the derivative.

### Definition
```
J = ∂f/∂x =
    [∂f₁/∂x₁  ∂f₁/∂x₂  ···  ∂f₁/∂xₙ]
    [∂f₂/∂x₁  ∂f₂/∂x₂  ···  ∂f₂/∂xₙ]
    [    ⋮          ⋮      ⋱      ⋮   ]
    [∂fₘ/∂x₁  ∂fₘ/∂x₂  ···  ∂fₘ/∂xₙ]
```

- Shape: (m × n)
- Reduces to gradient (column vector) when m = 1
- Reduces to single derivative when m = n = 1

### Chain Rule with Jacobians
```
∂z/∂x = (∂z/∂y) · (∂y/∂x)   ← matrix multiplication of Jacobians
```

### Jacobian Determinant
```
|J|  (scalar)
```
Measures how much a transformation stretches or squishes volume at a point.

**AI uses:**
- **Normalizing flows:** Change of variables in probability — det|J| appears in the density transformation formula
- **Softmax gradient:** Jacobian of softmax is non-trivial (off-diagonal terms)
- **Backprop through vector layers:** Each layer's gradient is a Jacobian-vector product

### Vector-Jacobian Product (VJP)
In reverse-mode autodiff (used in PyTorch/JAX):
```
vᵀ · J   — cheap to compute via backprop
```
This is what `.backward()` actually computes.

---

## 10. Taylor Series & Approximation

### Taylor Series
Any smooth function can be approximated as an infinite polynomial around point a:

```
f(x) = f(a) + f'(a)(x−a) + f''(a)(x−a)²/2! + f'''(a)(x−a)³/3! + ···
     = Σₙ₌₀^∞ fⁿ(a)(x−a)ⁿ / n!
```

### First-Order (Linear) Approximation
```
f(x) ≈ f(a) + f'(a)(x−a)
```
**AI use:** Foundation of gradient descent — the loss surface is locally approximated as linear.

### Second-Order Approximation
```
f(x) ≈ f(a) + f'(a)(x−a) + ½f''(a)(x−a)²
```
**AI use:** Newton's method, trust-region optimization, natural gradient.

### Multivariate Taylor Expansion
```
f(x) ≈ f(a) + ∇f(a)ᵀ(x−a) + ½(x−a)ᵀ H(a) (x−a) + ···
```

### Common Taylor Series
```
eˣ     = 1 + x + x²/2! + x³/3! + ···
ln(1+x) = x − x²/2 + x³/3 − ···        (|x| < 1)
sin(x) = x − x³/3! + x⁵/5! − ···
cos(x) = 1 − x²/2! + x⁴/4! − ···
1/(1−x) = 1 + x + x² + x³ + ···        (|x| < 1)
σ(x)   ≈ 0.5 + 0.25x                   (near x = 0)
```

### AI Applications
- Gradient descent derivation
- Analyzing learning rate sensitivity
- Warm-up and momentum analysis
- Understanding why small learning rates are safe

---

## 11. Multivariable Optimization

### Critical Points
At a critical point, the gradient is zero:
```
∇f(x*) = 0
```

Types of critical points:
- **Local minimum:** f(x*) ≤ f(x) in a neighborhood
- **Local maximum:** f(x*) ≥ f(x) in a neighborhood
- **Saddle point:** Neither min nor max

### Second-Order Conditions
```
If ∇f(x*) = 0 and H(x*) is positive definite → local minimum
If ∇f(x*) = 0 and H(x*) is negative definite → local maximum
If ∇f(x*) = 0 and H(x*) is indefinite → saddle point
```

### Convexity
A function f is **convex** if:
```
f(λx + (1−λ)y) ≤ λf(x) + (1−λ)f(y)   for λ ∈ [0,1]
```

**Properties of convex functions:**
- Any local minimum is a global minimum
- f is convex iff H(x) is positive semi-definite everywhere
- Examples: MSE loss, L2 norm, log-sum-exp

**AI context:** Most deep learning loss surfaces are **non-convex** — they have many local minima, saddle points, and flat regions. Yet gradient descent often finds good solutions.

### Constrained Optimization — Lagrange Multipliers
Minimize f(x) subject to g(x) = 0:
```
∇f(x) = λ ∇g(x)
```
The Lagrangian: L(x, λ) = f(x) − λ·g(x)

**AI use:** Deriving SVM optimization, constrained policy optimization in RL (PPO).

---

## 12. Gradient Descent & Variants

### Gradient Descent
The fundamental optimization algorithm in AI:
```
θ ← θ − α · ∇_θ L(θ)
```
- θ: parameters (weights)
- α: learning rate (step size)
- ∇_θ L: gradient of loss with respect to parameters

### Why Negative Gradient?
- ∇L points toward steepest **increase**
- We want to **decrease** loss
- So we step in the **opposite** direction: −∇L

### Learning Rate Effects
```
α too large  → overshoots minimum, diverges
α too small  → very slow convergence
α just right → smooth, stable convergence
```

### Variants of Gradient Descent

#### Batch Gradient Descent
```
∇L = (1/N) Σᵢ ∇L(xᵢ, yᵢ)   (all N training examples)
```
- Stable, accurate gradient estimate
- Very slow for large datasets

#### Stochastic Gradient Descent (SGD)
```
∇L ≈ ∇L(xᵢ, yᵢ)   (single random example)
```
- Fast updates, noisy gradient
- Noise can help escape local minima

#### Mini-Batch SGD (Most Common in Practice)
```
∇L ≈ (1/B) Σᵢ∈batch ∇L(xᵢ, yᵢ)   (B examples per batch)
```
- Balance between stability and speed
- B = 32, 64, 128, 256 are common choices

### Momentum
Accelerates convergence, reduces oscillation:
```
v ← βv − α·∇L
θ ← θ + v
```
- β = 0.9 typical
- v accumulates gradient history (exponential moving average)

### RMSProp
Adapts learning rate per parameter:
```
s ← ρs + (1−ρ)(∇L)²
θ ← θ − (α / √(s + ε)) · ∇L
```

### Adam (Adaptive Moment Estimation)
Combines momentum + RMSProp — the **most widely used optimizer** in deep learning:
```
m ← β₁m + (1−β₁)∇L           (1st moment — mean)
v ← β₂v + (1−β₂)(∇L)²        (2nd moment — variance)

m̂ = m / (1−β₁ᵗ)              (bias correction)
v̂ = v / (1−β₂ᵗ)

θ ← θ − α · m̂ / (√v̂ + ε)
```

Typical hyperparameters: α=0.001, β₁=0.9, β₂=0.999, ε=10⁻⁸

### Learning Rate Schedules
```
Step decay:      α = α₀ · γ^(epoch/drop_every)
Exponential:     α = α₀ · e^(−kt)
Cosine annealing: α = α_min + ½(α_max − α_min)(1 + cos(πt/T))
Warmup:          Linear increase for first few steps, then decay
```

---

## 13. Integrals — Concepts for AI

### Definite Integral
```
∫ₐᵇ f(x) dx = lim_{n→∞} Σᵢ f(xᵢ) Δx
```
Geometric interpretation: Area under the curve of f(x) from a to b.

### Fundamental Theorem of Calculus
```
d/dx [∫ₐˣ f(t) dt] = f(x)
```
Differentiation and integration are inverse operations.

### Key Integral Rules
```
∫ xⁿ dx = xⁿ⁺¹/(n+1) + C
∫ eˣ dx = eˣ + C
∫ 1/x dx = ln|x| + C
∫ sin(x) dx = −cos(x) + C
∫ cos(x) dx = sin(x) + C
```

### Integration Techniques

**Substitution (u-sub):**
```
∫ f(g(x))g'(x) dx = ∫ f(u) du    where u = g(x)
```

**Integration by parts:**
```
∫ u dv = uv − ∫ v du
```

### Improper Integrals (Important for Probability)
```
∫₋∞^∞ e^(−x²) dx = √π       (Gaussian integral — fundamental!)
∫₋∞^∞ N(μ,σ²) dx = 1        (all PDFs integrate to 1)
```

### AI Uses of Integration

**Expected values:**
```
E[X] = ∫ x · f(x) dx
```

**Normalizing distributions:**
```
∫₋∞^∞ f(x) dx = 1   (required for any valid PDF)
```

**Evidence in Bayesian inference (intractable in high dimensions):**
```
P(E) = ∫ P(E|H) P(H) dH
```
This integral is often **intractable** → requires approximation methods like MCMC or variational inference.

**Normalizing flows — change of variables:**
```
p_x(x) = p_z(f⁻¹(x)) · |det J_{f⁻¹}(x)|
```

---

## 14. Calculus in Neural Networks (End-to-End)

### Forward Pass
Information flows from input to output, computing predictions:
```
z⁽¹⁾ = W⁽¹⁾x + b⁽¹⁾
a⁽¹⁾ = σ(z⁽¹⁾)
z⁽²⁾ = W⁽²⁾a⁽¹⁾ + b⁽²⁾
ŷ    = softmax(z⁽²⁾)
L    = CrossEntropy(ŷ, y)
```

### Backward Pass (Backpropagation)
Chain rule applied right-to-left through the computational graph:

```
Step 1: ∂L/∂ŷ      (loss derivative)

Step 2: ∂L/∂z⁽²⁾ = ∂L/∂ŷ · ∂ŷ/∂z⁽²⁾

Step 3: ∂L/∂W⁽²⁾ = ∂L/∂z⁽²⁾ · (a⁽¹⁾)ᵀ
        ∂L/∂b⁽²⁾ = ∂L/∂z⁽²⁾

Step 4: ∂L/∂a⁽¹⁾ = (W⁽²⁾)ᵀ · ∂L/∂z⁽²⁾

Step 5: ∂L/∂z⁽¹⁾ = ∂L/∂a⁽¹⁾ ⊙ σ'(z⁽¹⁾)    (⊙ = element-wise)

Step 6: ∂L/∂W⁽¹⁾ = ∂L/∂z⁽¹⁾ · xᵀ
        ∂L/∂b⁽¹⁾ = ∂L/∂z⁽¹⁾
```

### Weight Update (Gradient Descent)
```
W⁽ˡ⁾ ← W⁽ˡ⁾ − α · ∂L/∂W⁽ˡ⁾
b⁽ˡ⁾ ← b⁽ˡ⁾ − α · ∂L/∂b⁽ˡ⁾
```

### Vanishing & Exploding Gradients
When networks are deep, chain rule multiplies many local gradients:
```
∂L/∂W⁽¹⁾ = ∂L/∂z⁽ᴸ⁾ · ∏ₗ (∂z⁽ˡ⁺¹⁾/∂z⁽ˡ⁾)
```

If each factor < 1: product → 0 (**vanishing gradient** — early layers don't learn)

If each factor > 1: product → ∞ (**exploding gradient** — training diverges)

**Solutions:**
- ReLU activation (avoids saturation)
- Batch normalization (normalizes activations)
- Residual connections / skip connections
- Gradient clipping (for exploding)
- Careful weight initialization (Xavier, He)

---

## 15. Activation Functions & Their Derivatives

Activation function derivatives are computed millions of times during backprop.

### Sigmoid
```
σ(x) = 1 / (1 + e⁻ˣ)

σ'(x) = σ(x) · (1 − σ(x))

Range: (0, 1)
Problem: Saturates for |x| large → vanishing gradient
Use: Binary classification output
```

### Tanh
```
tanh(x) = (eˣ − e⁻ˣ) / (eˣ + e⁻ˣ)

tanh'(x) = 1 − tanh²(x)

Range: (−1, 1)
Advantage: Zero-centered (better than sigmoid)
Problem: Still saturates
```

### ReLU (Rectified Linear Unit) — Most Common
```
ReLU(x) = max(0, x)

ReLU'(x) = 1 if x > 0, else 0

Range: [0, ∞)
Advantage: No vanishing gradient for x > 0, computationally cheap
Problem: "Dead ReLU" — neurons stuck at 0 if x always ≤ 0
```

### Leaky ReLU
```
f(x) = x  if x > 0,  else αx  (α ≈ 0.01)

f'(x) = 1  if x > 0,  else α

Fixes dead ReLU problem
```

### ELU (Exponential Linear Unit)
```
f(x) = x       if x > 0
f(x) = α(eˣ−1) if x ≤ 0

f'(x) = 1      if x > 0
f'(x) = f(x)+α if x ≤ 0
```

### GELU (Gaussian Error Linear Unit) — Used in Transformers
```
GELU(x) = x · Φ(x)   where Φ is the standard normal CDF

GELU'(x) = Φ(x) + x · φ(x)   where φ is the normal PDF

Used in: BERT, GPT, all modern transformers
```

### Softmax
```
softmax(z)ₖ = exp(zₖ) / Σⱼ exp(zⱼ)

∂softmax(z)ₖ/∂zⱼ = softmax(z)ₖ · (δₖⱼ − softmax(z)ⱼ)    (Jacobian, not scalar)
```

### Swish
```
swish(x) = x · σ(x)
swish'(x) = σ(x) + x · σ(x)(1 − σ(x))

Smooth, non-monotonic, performs well in deep networks
```

---

## 16. Automatic Differentiation

Modern AI frameworks (PyTorch, TensorFlow, JAX) compute gradients automatically via **autodiff** — not symbolic math, not numerical differences.

### Three Approaches

#### Symbolic Differentiation
- Apply differentiation rules symbolically (like Wolfram Alpha)
- Produces closed-form expressions
- Can be slow and produce complex expressions

#### Numerical Differentiation
```
f'(x) ≈ [f(x + h) − f(x)] / h     (forward difference)
f'(x) ≈ [f(x+h) − f(x−h)] / (2h) (central difference, more accurate)
```
- Easy to implement
- Subject to floating-point error
- O(n) extra forward passes for n parameters — impractical for millions of weights

#### Automatic Differentiation (Autodiff)
- Records all operations in a **computational graph**
- Applies chain rule exactly through the graph
- No approximation error (unlike numerical)
- No expression explosion (unlike symbolic)
- O(1) extra cost relative to forward pass

### Two Modes

**Forward mode (Jacobian-vector product):**
- Propagates derivatives forward alongside values
- Efficient when inputs ≪ outputs

**Reverse mode (Vector-Jacobian product = backprop):**
- Propagates derivatives backward from outputs to inputs
- Efficient when outputs ≪ inputs
- Neural networks: 1 loss output, millions of weight inputs → **reverse mode wins**

### How PyTorch Implements It
```python
x = torch.tensor(2.0, requires_grad=True)
y = x**3 + 2*x        # builds computational graph
y.backward()           # reverse-mode autodiff (backprop)
print(x.grad)          # ∂y/∂x = 3x² + 2 = 14 at x=2
```

The graph records: `x → x³ → x³ + 2x → y`, then traverses it in reverse with chain rule.

### Gradient Tape (TensorFlow / JAX)
```python
with tf.GradientTape() as tape:
    y = x**3 + 2*x
grad = tape.gradient(y, x)   # same result
```

---

## 17. Quick Reference Cheat Sheet

### Essential Derivative Rules
```
Power:     d/dx[xⁿ] = nxⁿ⁻¹
Chain:     d/dx[f(g(x))] = f'(g(x))·g'(x)
Product:   d/dx[fg] = f'g + fg'
Quotient:  d/dx[f/g] = (f'g − fg')/g²
Log:       d/dx[ln f(x)] = f'(x)/f(x)
Exp:       d/dx[e^f(x)] = e^f(x)·f'(x)
```

### Key Activation Derivatives
```
σ'(x)    = σ(x)(1 − σ(x))
tanh'(x) = 1 − tanh²(x)
ReLU'(x) = 1 if x > 0, else 0
```

### Gradient Descent Family
```
SGD:       θ ← θ − α·∇L
Momentum:  v ← βv − α·∇L;  θ ← θ + v
RMSProp:   s ← ρs + (1−ρ)(∇L)²;  θ ← θ − α·∇L/√(s+ε)
Adam:      m̂, v̂ bias-corrected moments;  θ ← θ − α·m̂/√(v̂+ε)
```

### Backprop Summary
```
Forward:   compute and store all zˡ, aˡ
Backward:  δᴸ = ∂L/∂zᴸ
           δˡ = ((Wˡ⁺¹)ᵀδˡ⁺¹) ⊙ σ'(zˡ)
Gradients: ∂L/∂Wˡ = δˡ(aˡ⁻¹)ᵀ
           ∂L/∂bˡ = δˡ
```

### Hessian Quick Guide
```
H positive definite  → local minimum
H negative definite  → local maximum
H indefinite         → saddle point
```

### Loss ↔ Calculus Connection
```
MSE loss:           L = (1/n)Σ(ŷ−y)²   → ∂L/∂ŷ = (2/n)(ŷ−y)
Cross-entropy loss: L = −Σ y log(ŷ)    → ∂L/∂ŷ = −y/ŷ
With softmax:       ∂L/∂z = ŷ − y      (elegant closed form)
```

---

## Study Roadmap

```
Week 1:  Sections 1–4    → Limits, derivatives, differentiation rules
Week 2:  Sections 5–7    → Partial derivatives, gradients, chain rule
Week 3:  Sections 8–10   → Hessian, Jacobian, Taylor series
Week 4:  Sections 11–12  → Optimization, gradient descent variants
Week 5:  Sections 13–14  → Integrals for AI, full backprop walkthrough
Week 6:  Sections 15–16  → Activation functions, automatic differentiation
```

### Recommended Practice
- Derive the gradient of MSE loss by hand
- Implement backpropagation for a 2-layer network from scratch in NumPy
- Plot sigmoid, tanh, ReLU and their derivatives side-by-side
- Implement gradient descent, momentum, and Adam from scratch
- Use PyTorch's `autograd` and verify gradients numerically
- Derive the softmax + cross-entropy combined gradient (ŷ − y)

---

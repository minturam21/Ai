# Probability Theory for AI — Complete Notes

> **Goal:** Build a solid, AI-ready understanding of probability theory — from first principles to the concepts that power modern machine learning.

---

## Table of Contents

1. [Why Probability Matters in AI](#1-why-probability-matters-in-ai)
2. [Sample Space, Events & Axioms](#2-sample-space-events--axioms)
3. [Types of Probability](#3-types-of-probability)
4. [Conditional Probability](#4-conditional-probability)
5. [Bayes' Theorem](#5-bayes-theorem)
6. [Independence](#6-independence)
7. [Random Variables](#7-random-variables)
8. [Probability Distributions](#8-probability-distributions)
   - 8.1 [Discrete Distributions](#81-discrete-distributions)
   - 8.2 [Continuous Distributions](#82-continuous-distributions)
9. [Expectation, Variance & Moments](#9-expectation-variance--moments)
10. [Joint, Marginal & Conditional Distributions](#10-joint-marginal--conditional-distributions)
11. [Important Theorems](#11-important-theorems)
12. [Maximum Likelihood Estimation (MLE)](#12-maximum-likelihood-estimation-mle)
13. [Maximum A Posteriori (MAP) Estimation](#13-maximum-a-posteriori-map-estimation)
14. [Information Theory Basics](#14-information-theory-basics)
15. [Probabilistic Graphical Models (Intro)](#15-probabilistic-graphical-models-intro)
16. [Probability in Neural Networks](#16-probability-in-neural-networks)
17. [Quick Reference Cheat Sheet](#17-quick-reference-cheat-sheet)

---

## 1. Why Probability Matters in AI

AI systems rarely operate on certain facts — they deal with **noisy data**, **missing information**, and **uncertain predictions**. Probability provides the mathematical language to reason under uncertainty.

| AI Task | Probability Concept Used |
|---|---|
| Classification | Posterior probability P(class \| input) |
| Language models | P(next word \| context) |
| Reinforcement learning | Stochastic policies, reward distributions |
| Bayesian inference | Prior × Likelihood → Posterior |
| Generative models (VAE, GAN) | Latent variable distributions |
| Anomaly detection | Likelihood under a learned distribution |

**Core idea:** Instead of predicting a single answer, AI models learn a *probability distribution* over possible answers.

---

## 2. Sample Space, Events & Axioms

### Sample Space (Ω)

The **sample space** is the set of all possible outcomes of a random experiment.

- Coin flip: Ω = {H, T}
- Die roll: Ω = {1, 2, 3, 4, 5, 6}
- Pixel intensity: Ω = {0, 1, 2, ..., 255}

### Events

An **event** A is a subset of Ω (a collection of outcomes).

- "Rolling an even number": A = {2, 4, 6}

### Kolmogorov's Axioms

For any event A in sample space Ω, probability P must satisfy:

1. **Non-negativity:** P(A) ≥ 0
2. **Normalization:** P(Ω) = 1
3. **Additivity:** If A ∩ B = ∅, then P(A ∪ B) = P(A) + P(B)

### Derived Rules

```
P(Aᶜ) = 1 − P(A)                    (complement)
P(∅) = 0                             (impossible event)
P(A ∪ B) = P(A) + P(B) − P(A ∩ B)  (inclusion-exclusion)
0 ≤ P(A) ≤ 1                         (bounded)
```

---

## 3. Types of Probability

### Frequentist Probability
- Probability = long-run frequency of an event
- Example: A fair coin lands heads 50% of the time over many flips
- **Used in:** classical statistics, hypothesis testing

### Bayesian Probability
- Probability = degree of belief, updated with evidence
- Example: P(model is correct | training data)
- **Used in:** Bayesian neural networks, probabilistic inference

### AI perspective
Most modern AI uses **both**: frequentist methods for training data, Bayesian thinking for uncertainty and model selection.

---

## 4. Conditional Probability

**Definition:** The probability of event A *given* that event B has occurred.

```
P(A | B) = P(A ∩ B) / P(B),   where P(B) > 0
```

### Example
- P(spam | contains "free") = proportion of "free" emails that are spam

### Multiplication Rule
```
P(A ∩ B) = P(A | B) · P(B) = P(B | A) · P(A)
```

### Law of Total Probability
If {B₁, B₂, ..., Bₙ} partitions Ω:
```
P(A) = Σᵢ P(A | Bᵢ) · P(Bᵢ)
```

**AI use case:** Computing the marginal probability of an output by summing over all possible hidden states or latent variables.

---

## 5. Bayes' Theorem

The single most important theorem in probabilistic AI.

```
P(H | E) = [ P(E | H) · P(H) ] / P(E)
```

| Term | Name | Meaning |
|---|---|---|
| P(H) | Prior | Belief before seeing evidence |
| P(E \| H) | Likelihood | How probable is E if H is true |
| P(E) | Evidence / Marginal | Normalizing constant |
| P(H \| E) | Posterior | Updated belief after evidence |

### Expanded Form (using total probability)
```
P(H | E) = P(E | H) · P(H) / Σⱼ P(E | Hⱼ) · P(Hⱼ)
```

### Concrete Example — Medical Diagnosis
- Disease prevalence: P(D) = 0.01
- Test sensitivity: P(+|D) = 0.95
- False positive rate: P(+|¬D) = 0.05

```
P(D|+) = (0.95 × 0.01) / [(0.95 × 0.01) + (0.05 × 0.99)]
        = 0.0095 / 0.0590 ≈ 0.161
```
Only 16% chance of disease despite a 95% accurate test — because prevalence is low.

### AI Use Cases
- **Naive Bayes classifier:** P(class | features) ∝ P(features | class) · P(class)
- **Bayesian neural networks:** Update weight distributions from data
- **Spam filters:** P(spam | words)
- **LLMs:** Implicitly performing Bayesian inference over language

---

## 6. Independence

### Definition
Events A and B are **independent** if:
```
P(A ∩ B) = P(A) · P(B)
```
Equivalently: P(A | B) = P(A)  — knowing B tells you nothing about A.

### Conditional Independence
A and B are conditionally independent given C if:
```
P(A ∩ B | C) = P(A | C) · P(B | C)
```

**This is the foundation of Naive Bayes:** features are assumed conditionally independent given the class label.

### Pairwise vs. Mutual Independence
- Pairwise: every pair is independent
- Mutual: all subsets are independent (stronger condition)

---

## 7. Random Variables

A **random variable** (RV) X is a function that maps outcomes in Ω to real numbers.

```
X : Ω → ℝ
```

### Types

| Type | Description | Example |
|---|---|---|
| Discrete | Countable values | Number of words in a sentence |
| Continuous | Uncountably infinite values | Temperature, pixel intensity |

### Probability Mass Function (PMF) — Discrete
```
p(x) = P(X = x),   Σₓ p(x) = 1
```

### Probability Density Function (PDF) — Continuous
```
f(x) ≥ 0,   ∫₋∞^∞ f(x) dx = 1
P(a ≤ X ≤ b) = ∫ₐᵇ f(x) dx
```

### Cumulative Distribution Function (CDF)
```
F(x) = P(X ≤ x)
```
Works for both discrete and continuous RVs.

---

## 8. Probability Distributions

### 8.1 Discrete Distributions

#### Bernoulli Distribution
- Models a single binary outcome
- Parameter: p (probability of success)

```
P(X = 1) = p
P(X = 0) = 1 − p
Mean = p,  Variance = p(1 − p)
```

**AI use:** Binary classification output (logistic regression outputs a Bernoulli probability).

---

#### Binomial Distribution B(n, p)
- Number of successes in n independent Bernoulli trials

```
P(X = k) = C(n,k) · pᵏ · (1−p)^(n−k)
Mean = np,  Variance = np(1−p)
```

---

#### Categorical Distribution
- Generalization of Bernoulli to K classes
- Parameters: p₁, p₂, ..., pₖ where Σpᵢ = 1

```
P(X = k) = pₖ
```

**AI use:** Multi-class classification. Softmax outputs a categorical distribution.

---

#### Multinomial Distribution
- Generalization of Binomial to K categories, n trials

**AI use:** Language models — probability of word counts in a document.

---

#### Poisson Distribution Poisson(λ)
- Count of events in a fixed interval, with average rate λ

```
P(X = k) = (λᵏ e^−λ) / k!
Mean = λ,  Variance = λ
```

---

### 8.2 Continuous Distributions

#### Uniform Distribution U(a, b)
```
f(x) = 1/(b−a)  for x ∈ [a, b]
Mean = (a+b)/2,  Variance = (b−a)²/12
```

**AI use:** Weight initialization, random sampling.

---

#### Gaussian (Normal) Distribution N(μ, σ²)

The most important distribution in all of statistics and AI.

```
f(x) = (1 / √(2πσ²)) · exp(−(x−μ)² / (2σ²))

Mean = μ
Variance = σ²
Standard deviation = σ
```

**Properties:**
- Symmetric, bell-shaped curve
- ~68% of data within ±1σ, ~95% within ±2σ, ~99.7% within ±3σ
- Sum of independent Gaussians is Gaussian

**AI uses:**
- Weight initialization (He, Xavier)
- Gaussian Naive Bayes
- Variational Autoencoders (VAE latent space)
- Gaussian Processes
- Noise modeling in regression

---

#### Standard Normal Distribution N(0, 1)
Standardize any Gaussian: `z = (x − μ) / σ`

---

#### Multivariate Gaussian N(μ, Σ)
```
f(x) = (1 / ((2π)^(d/2) |Σ|^(1/2))) · exp(−½(x−μ)ᵀΣ⁻¹(x−μ))

μ = mean vector (d-dimensional)
Σ = covariance matrix (d×d, positive semi-definite)
```

**AI use:** Gaussian mixture models, VAE, Gaussian processes, Kalman filters.

---

#### Exponential Distribution Exp(λ)
```
f(x) = λe^(−λx)  for x ≥ 0
Mean = 1/λ,  Variance = 1/λ²
```

---

#### Beta Distribution Beta(α, β)
```
f(x) = x^(α−1) · (1−x)^(β−1) / B(α, β)   for x ∈ [0, 1]
Mean = α/(α+β)
```

**AI use:** Prior distribution over probabilities (conjugate prior for Bernoulli/Binomial).

---

#### Dirichlet Distribution Dir(α)
- Generalization of Beta to K dimensions
- Distribution over probability vectors (simplex)

**AI use:** Prior over categorical distributions, topic modeling (LDA).

---

## 9. Expectation, Variance & Moments

### Expectation (Mean)
```
E[X] = Σₓ x · p(x)              (discrete)
E[X] = ∫ x · f(x) dx            (continuous)
```

**Properties:**
```
E[aX + b] = aE[X] + b           (linearity)
E[X + Y] = E[X] + E[Y]          (always)
E[XY] = E[X]·E[Y]               (only if X, Y independent)
```

### Variance
```
Var(X) = E[(X − μ)²] = E[X²] − (E[X])²

Var(aX + b) = a² Var(X)
Var(X + Y) = Var(X) + Var(Y)    (if independent)
```

### Standard Deviation
```
σ = √Var(X)
```

### Covariance
```
Cov(X, Y) = E[(X − μₓ)(Y − μᵧ)] = E[XY] − E[X]E[Y]
```
- Cov > 0: X and Y increase together
- Cov < 0: X increases as Y decreases
- Cov = 0: uncorrelated (not necessarily independent!)

### Correlation
```
ρ(X, Y) = Cov(X, Y) / (σₓ · σᵧ),   ρ ∈ [−1, 1]
```

### Higher Moments
- **1st moment:** Mean E[X]
- **2nd central moment:** Variance E[(X−μ)²]
- **3rd standardized moment:** Skewness (asymmetry)
- **4th standardized moment:** Kurtosis (tail heaviness)

**AI relevance:** Batch normalization normalizes to zero mean and unit variance; skewness/kurtosis help diagnose data quality issues.

---

## 10. Joint, Marginal & Conditional Distributions

### Joint Distribution
```
p(x, y) = P(X = x, Y = y)       (discrete)
f(x, y)                          (continuous)
```

### Marginal Distribution
Obtained by summing/integrating out the other variable:
```
p(x) = Σᵧ p(x, y)               (discrete)
f(x) = ∫ f(x, y) dy             (continuous)
```

### Conditional Distribution
```
p(y | x) = p(x, y) / p(x)
```

### Chain Rule of Probability
```
P(A₁ ∩ A₂ ∩ ... ∩ Aₙ) = P(A₁) · P(A₂|A₁) · P(A₃|A₁,A₂) · ... · P(Aₙ|A₁,...,Aₙ₋₁)
```

**AI use:** This is literally how autoregressive language models work:
```
P(w₁, w₂, ..., wₙ) = P(w₁) · P(w₂|w₁) · P(w₃|w₁,w₂) · ...
```

---

## 11. Important Theorems

### Law of Large Numbers (LLN)
As sample size n → ∞, the sample mean converges to the true mean:
```
(1/n) Σᵢ Xᵢ → E[X]  as n → ∞
```

**AI use:** Justifies training on large datasets; mini-batch gradient estimates converge to true gradient.

---

### Central Limit Theorem (CLT)
The sum (or mean) of many independent, identically distributed RVs approaches a Gaussian, regardless of the original distribution:
```
√n · (X̄ − μ) / σ → N(0, 1)  as n → ∞
```

**AI use:**
- Justifies Gaussian assumptions in many models
- Explains why neural network weight distributions are often approximately Gaussian
- Underpins confidence intervals and hypothesis tests

---

### Jensen's Inequality
For a convex function φ:
```
φ(E[X]) ≤ E[φ(X)]
```
For concave functions (like log), the inequality reverses.

**AI use:** Derives the Evidence Lower Bound (ELBO) in VAEs:
```
log P(x) ≥ E[log P(x|z)] − KL(Q(z|x) || P(z))
```

---

## 12. Maximum Likelihood Estimation (MLE)

**Goal:** Find parameters θ that make the observed data most probable.

```
θ_MLE = argmax_θ P(data | θ)
       = argmax_θ L(θ)
```

### Log-Likelihood
Taking log (monotone transformation, makes products into sums):
```
θ_MLE = argmax_θ log L(θ) = argmax_θ Σᵢ log P(xᵢ | θ)
```

### Example — MLE for Gaussian
Given data x₁, ..., xₙ from N(μ, σ²):
```
μ_MLE = (1/n) Σᵢ xᵢ           (sample mean)
σ²_MLE = (1/n) Σᵢ (xᵢ − μ)²  (biased sample variance)
```

### MLE and Loss Functions
Minimizing **cross-entropy loss** = maximizing log-likelihood under a categorical distribution.

Minimizing **mean squared error (MSE)** = maximizing log-likelihood under a Gaussian noise model.

---

## 13. Maximum A Posteriori (MAP) Estimation

**Goal:** Find θ that maximizes the posterior — combines data likelihood with prior belief.

```
θ_MAP = argmax_θ P(θ | data)
       = argmax_θ [log P(data | θ) + log P(θ)]
```

### MLE vs MAP

| | MLE | MAP |
|---|---|---|
| Prior | No prior (or uniform) | Informative prior |
| Regularization | None | L2 reg = Gaussian prior; L1 reg = Laplace prior |
| Overfitting | More prone | Less prone (regularized) |

**Key insight:** L2 regularization (weight decay) in neural networks is MAP estimation with a Gaussian prior over weights.

---

## 14. Information Theory Basics

### Entropy
Measures the **average uncertainty** (unpredictability) in a distribution:
```
H(X) = −Σₓ p(x) log p(x)      (discrete)
H(X) = −∫ f(x) log f(x) dx    (continuous, called differential entropy)
```

- H = 0: perfectly predictable (no uncertainty)
- H = log K: maximum for K outcomes (uniform distribution)
- Units: bits (log base 2) or nats (natural log)

**AI use:** Decision tree splitting criteria (information gain = reduction in entropy).

---

### Cross-Entropy
Measures the average bits needed to encode data from distribution p using code optimized for q:
```
H(p, q) = −Σₓ p(x) log q(x)
```

**AI use:** The standard classification loss function.
```python
loss = -sum(y_true * log(y_pred))  # cross-entropy loss
```

---

### KL Divergence (Kullback-Leibler)
Measures how different distribution q is from reference distribution p:
```
KL(p || q) = Σₓ p(x) log(p(x) / q(x))
           = H(p, q) − H(p)
```

**Properties:**
- KL(p || q) ≥ 0 always
- KL(p || q) = 0 iff p = q
- **Asymmetric:** KL(p||q) ≠ KL(q||p)

**AI uses:**
- VAE regularization term: KL(Q(z|x) || P(z))
- Knowledge distillation
- Policy optimization in RL (PPO uses KL constraint)

---

### Mutual Information
Measures how much knowing X tells you about Y:
```
I(X; Y) = KL(P(X,Y) || P(X)P(Y))
         = H(X) − H(X|Y)
         = H(Y) − H(Y|X)
```

- I(X; Y) = 0: X and Y are independent
- Higher I: more information shared

**AI use:** Feature selection, representation learning, information bottleneck theory.

---

## 15. Probabilistic Graphical Models (Intro)

### Bayesian Networks (Directed)
- Directed Acyclic Graph (DAG) representing conditional independence
- Each node = random variable; each edge = direct probabilistic influence

```
Joint distribution factorizes as:
P(X₁, ..., Xₙ) = Π P(Xᵢ | Parents(Xᵢ))
```

**Example:** Naive Bayes is a Bayesian network where all features are independent given the class.

### Markov Random Fields (Undirected)
- Undirected graph; encode correlations without causal direction
- **AI use:** Image segmentation, CRFs in NLP

### Key Concepts
- **d-separation:** Determines conditional independence from graph structure
- **Belief propagation:** Efficient inference algorithm on graphs
- **Hidden Markov Model (HMM):** Sequential Bayesian network — used in speech recognition, NLP

---

## 16. Probability in Neural Networks

### Softmax — Categorical Distribution
Converts raw logits into a valid probability distribution:
```
P(y = k | x) = exp(zₖ) / Σⱼ exp(zⱼ)
```
Output sums to 1, each value in (0, 1).

### Sigmoid — Bernoulli Distribution
For binary classification:
```
P(y = 1 | x) = 1 / (1 + e^(−z)) = σ(z)
```

### Dropout as Bayesian Approximation
- Dropout at test time approximates a Bayesian ensemble of models
- Provides uncertainty estimates (Monte Carlo Dropout)

### Variational Autoencoders (VAE)
- Encoder: q_φ(z|x) ≈ P(z|x) — approximate posterior
- Decoder: P_θ(x|z) — likelihood
- Loss = Reconstruction loss + KL(q_φ(z|x) || P(z))

### Probabilistic Outputs in LLMs
Language models output a probability distribution over the vocabulary at each step:
```
P(wₜ | w₁, ..., wₜ₋₁) — via softmax over vocabulary
```
Temperature scaling adjusts sharpness:
```
P(wₜ) ∝ exp(zₜ / T)
```
- T < 1: more confident (sharper)
- T > 1: more uniform (creative)

---

## 17. Quick Reference Cheat Sheet

### Essential Formulas

```
Bayes' Theorem:        P(H|E) = P(E|H)·P(H) / P(E)
Total Probability:     P(A) = Σᵢ P(A|Bᵢ)·P(Bᵢ)
Expectation:           E[X] = Σ x·p(x)
Variance:              Var(X) = E[X²] − (E[X])²
Gaussian PDF:          f(x) = (1/√(2πσ²)) exp(−(x−μ)²/2σ²)
Entropy:               H(X) = −Σ p(x) log p(x)
Cross-Entropy:         H(p,q) = −Σ p(x) log q(x)
KL Divergence:         KL(p||q) = Σ p(x) log(p(x)/q(x))
MLE:                   θ* = argmax Σ log P(xᵢ|θ)
MAP:                   θ* = argmax [Σ log P(xᵢ|θ) + log P(θ)]
Softmax:               P(k) = exp(zₖ) / Σ exp(zⱼ)
```

### Key Distributions Summary

| Distribution | Parameters | Mean | Variance | AI Use |
|---|---|---|---|---|
| Bernoulli | p | p | p(1-p) | Binary classification |
| Categorical | p₁…pₖ | — | — | Multi-class output |
| Gaussian N(μ,σ²) | μ, σ² | μ | σ² | Everything |
| Multivariate Gaussian | μ, Σ | μ | Σ | VAEs, GMMs |
| Beta(α,β) | α, β | α/(α+β) | — | Prior over probabilities |
| Dirichlet | α | αᵢ/Σαⱼ | — | Prior over categoricals |
| Exponential | λ | 1/λ | 1/λ² | Time-to-event |

### Loss Function ↔ Probabilistic Interpretation

| Loss Function | Probabilistic Interpretation |
|---|---|
| Cross-Entropy | MLE under categorical distribution |
| MSE | MLE under Gaussian noise model |
| Binary Cross-Entropy | MLE under Bernoulli distribution |
| L2 Regularization | MAP with Gaussian prior |
| L1 Regularization | MAP with Laplace prior |
| KL term in VAE | Regularize posterior toward prior |

---

## Study Roadmap

```
Week 1:  Sections 1–4   → Foundations, conditional probability
Week 2:  Sections 5–6   → Bayes' theorem, independence
Week 3:  Sections 7–8   → Random variables, distributions
Week 4:  Sections 9–10  → Expectation, joint distributions
Week 5:  Sections 11–13 → Theorems, MLE, MAP
Week 6:  Sections 14–16 → Information theory, neural nets
```

### Recommended Practice
- Implement Gaussian, Bernoulli, and Categorical distributions from scratch in NumPy
- Derive MLE for Gaussian parameters by hand
- Code Naive Bayes from scratch using Bayes' theorem
- Compute KL divergence between two Gaussians analytically
- Visualize how temperature scaling changes a softmax distribution

---

*These notes cover all probability concepts required for ML/DL, NLP, computer vision, and probabilistic AI. Cross-reference with linear algebra and calculus notes for a complete AI mathematics foundation.*

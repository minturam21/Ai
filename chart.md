# AI Learning Roadmap
```mermaid
graph TD
A[Math Foundations] --> B[Programming]
B --> C[Machine Learning]
C --> D[Deep Learning]
D --> E[NLP & Computer Vision]
E --> F[MLOps]
F --> G[AI Projects]
G --> H[Job Ready]
```
# Roadmap Timeline
```mermaid
gantt
    title AI Preparation Timeline
    dateFormat  YYYY-MM-DD
    section Foundation
    Mathematics           :a1, 2026-01-01, 60d
    Python Programming    :a2, after a1, 45d

    section Core AI
    Machine Learning      :b1, after a2, 60d
    Deep Learning         :b2, after b1, 60d

    section Specialization
    NLP                   :c1, after b2, 45d
    Computer Vision       :c2, after c1, 45d

    section Engineering
    MLOps                 :d1, after c2, 45d
    AI Projects           :d2, after d1, 60d
```
# Mathematics for AI

* Topics

* Linear Algebra
* Probability
* Statistics
* Calculus
* Optimization

* Key Concepts

* Vectors & matrices
* Eigenvalues & eigenvectors
* Probability distributions
* Gradient descent
* Partial derivatives

 # Programming for AI

- Language focus:

```Python
Python
```

# Libraries:

* NumPy
* Pandas
* Matplotlib
* Scikit-learn

# Skills to master:

* data preprocessing
* vectorized computation
* debugging
* dataset manipulation
  
# Machine Learning
Supervised Learning
Algorithms
- Linear Regression
- Logistic Regression
- Decision Trees
- Random Forest
- Gradient Boosting
- Support Vector Machines

Unsupervised Learning
- K-Means
- Hierarchical clustering
- PCA
- Dimensionality reduction

# ML Workflow Animation
```mermaid
flowchart LR
    A[Data] --> B[Preprocessing]
    B --> C[Training]
    C --> D[Evaluation]
    D --> E[Deployment]
```

# Deep Learning
Frameworks
```python
PyTorch
TensorFlow
```
Neural Network Concepts
- Perceptron
- Activation functions
- Backpropagation

Architectures
- CNN
- RNN
- Transformers

  # Neural Network Visualization
  ```mermaid
  flowchart LR
    A[Input Layer] --> B[Hidden Layer]
    B --> C[Hidden Layer]
    C --> D[Output Layer]
  ```

  # Natural Language Processing

Topics
- Tokenization
- Word Embeddings
- Transformers
- BERT / GPT models
- Text classification

Example applications
- chatbots
- sentiment analysis
- document search

# Computer Vision

Topics
- Image preprocessing
- CNN architectures
- Object detection
- Image segmentation

Libraries
```
OpenCV
PyTorch
TensorFlow
```
# MLOps
Skills
- model deployment
- Docker
- FastAPI
- CI/CD
- cloud platforms

Tools
```
Docker
Kubernetes
MLflow
Airflow
AWS / GCP
```
# Data Engineering

Important for real AI systems.

Skills

- SQL
- ETL pipelines
- Spark basics
- large dataset handling

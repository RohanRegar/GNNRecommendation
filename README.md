# Graph Neural Network-Based Product Recommendation System

A sophisticated recommendation engine leveraging Graph Neural Networks (GNNs) to predict user preferences for Amazon products. This implementation demonstrates the power of graph-based deep learning approaches for collaborative filtering and recommendation tasks.

---

## 🎯 Project Overview

This project implements a state-of-the-art recommendation system using heterogeneous graph neural networks. Unlike traditional matrix factorization methods, this approach models users and products as nodes in a graph, with relationships represented as edges, enabling the capture of complex interaction patterns.

### Key Features

**Graph-Based Architecture**

-   Heterogeneous graph with user and product nodes
-   Multiple edge types: user-product reviews and product-product "also bought" relationships
-   Captures both user preferences and product associations

**Advanced Neural Network Models**

-   **GraphSAGE**: Inductive learning with neighborhood sampling
-   **Graph Attention Networks (GAT)**: Attention-based aggregation for weighted neighbor information
-   Custom encoder-decoder architecture for rating prediction

**Rich Feature Engineering**

-   Product embeddings generated using state-of-the-art sentence transformers (all-MiniLM-L6-v2)
-   One-hot encoded product categories
-   Dynamic user representations learned through graph convolutions

**Model Interpretability**

-   Integration with Captum library for explainable AI
-   Visualizations of attention weights and feature importance

---

## 📊 Dataset

-   **Source**: Amazon Product Reviews (2018) - Software Category
-   **Preprocessing**: 5-core filtering (users and products with minimum 5 interactions)
-   **Statistics**:
    -   1,826 unique users
    -   21,639 software products
    -   Multiple thousands of review interactions
    -   Product metadata including descriptions, categories, and co-purchase information

---

## 🔧 Technical Implementation

### Architecture Components

1. **Node Feature Generation**

    - Products: 384-dimensional text embeddings + categorical features
    - Users: Learned embeddings through graph propagation

2. **Graph Convolutional Layers**

    - Layer 1: Neighborhood aggregation with message passing
    - Layer 2: Higher-order graph convolutions for deeper representations

3. **Edge Prediction Decoder**
    - Concatenates user and product embeddings
    - Multi-layer perceptron for rating prediction
    - Output: Rating scores (1-5 scale)

### Evaluation Metrics

-   Root Mean Squared Error (RMSE) on test set
-   Comparison with baseline models (KNN, SVD)

---

## 🚀 Setup and Usage

### Requirements

-   **Python 3.9 - 3.11** (Python 3.12+ not recommended due to TensorFlow compatibility issues)
-   CUDA-capable GPU (recommended for training)
-   8GB+ RAM

### Installation Steps

1. **Clone the repository**

    ```bash
    git clone <your-repo-url>
    cd amazonrecommendation
    ```

2. **Create virtual environment** (recommended)

    ```bash
    # Using conda
    conda create -n gnn_recommender python=3.11
    conda activate gnn_recommender

    # OR using venv
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```

3. **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the notebook**

    Open `ARS.ipynb` in Jupyter Notebook or JupyterLab and execute cells sequentially.

    ```bash
    jupyter notebook ARS.ipynb
    ```

### Important Notes

-   Dataset files are included in the `data/` directory
-   No need to download data separately
-   First run may take time for embedding generation
-   Model checkpoints can be saved for inference

---

## 📁 Project Structure

```
amazonrecommendation/
├── ARS.ipynb              # Main implementation notebook
├── data/
│   ├── Software_5.json    # User reviews (5-core)
│   ├── meta_Software.json # Product metadata
│   └── Software.csv       # Additional data
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

---

## 🛠️ Technologies Used

-   **Deep Learning**: PyTorch 2.1+, PyTorch Geometric
-   **NLP**: Hugging Face Transformers, Sentence Transformers
-   **Data Processing**: Pandas, NumPy, Scikit-learn
-   **Visualization**: Matplotlib
-   **Explainability**: Captum
-   **Notebook**: Jupyter

---

## 📈 Model Performance

The GraphSAGE model achieves competitive performance with:

-   Test RMSE: ~1.21
-   Improved performance over traditional collaborative filtering
-   Better cold-start handling through inductive learning

---

## 🔍 Key Insights

1. **Graph Structure Matters**: Products connected via "also bought" relationships show correlated rating patterns
2. **Attention Mechanisms**: GAT layers help identify which neighbor nodes are most relevant
3. **Rich Features**: Combining text embeddings with categorical data improves predictions
4. **Scalability**: Inductive learning enables predictions for new users/products

---


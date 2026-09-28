# Simple RAG Pipeline with FAISS and Hugging Face

A practical implementation of a simple Retrieval-Augmented Generation (RAG) pipeline using Sentence Transformers, FAISS, and Hugging Face Transformers.

This project demonstrates how a language model can use retrieved information from a custom knowledge base to generate more context-aware responses. The notebook also compares a standard language model response with a response generated using retrieved context.

## Overview

Large language models can generate useful responses, but they may not always have access to specific or relevant information required to answer a question.

Retrieval-Augmented Generation addresses this by combining two main components:

1. Retrieval: Find relevant information from a knowledge base.
2. Generation: Provide the retrieved information to a language model as context for generating the final answer.

In this project, a small collection of AI-related documents is converted into numerical embeddings using a Sentence Transformer model. These embeddings are stored in a FAISS index, which is then used to retrieve the most relevant documents for a given query.

The retrieved documents are passed to a Hugging Face text generation model to produce a context-aware response.

## RAG Pipeline

The complete workflow implemented in the notebook is:

```text
Documents
    |
    v
Sentence Transformer
    |
    v
Document Embeddings
    |
    v
FAISS Index
    |
    v
User Query
    |
    v
Query Embedding
    |
    v
Similarity Search
    |
    v
Top Relevant Documents
    |
    v
Retrieved Context
    |
    v
FLAN-T5
    |
    v
RAG Response
```

## Project Components

### 1. Knowledge Base

The project uses a small collection of AI-related documents covering topics such as:

* Deep learning
* Generative AI
* AI optimization
* Natural language processing
* Computer vision
* Reinforcement learning
* Transfer learning
* Attention mechanisms

These documents act as the knowledge base for the retrieval process.

### 2. Embedding Generation

The project uses the following Sentence Transformers model:

```text
paraphrase-MiniLM-L6-v2
```

Each document is converted into a numerical vector representation called an embedding.

Embeddings allow semantically similar pieces of text to be represented by vectors that are close to each other in the embedding space.

### 3. FAISS Vector Search

FAISS is used to store and search the document embeddings.

The notebook creates the following FAISS index:

```python
faiss.IndexFlatL2(dimension)
```

The index uses L2 distance to measure the similarity between the query embedding and document embeddings.

For each query, the system retrieves the top three most relevant documents.

### 4. Query Processing

The user's question is converted into an embedding using the same Sentence Transformer model used for the documents.

The query embedding is then compared against the vectors stored in FAISS.

This allows the system to retrieve documents based on semantic similarity rather than simply matching exact keywords.

### 5. Context Construction

The retrieved documents are combined into a single context string.

This context is then included in the prompt sent to the language model.

The general prompt structure is:

```text
Use the following context to answer the question.

Context:
Retrieved documents

Question:
User query

Answer:
```

### 6. Text Generation

The project uses the Hugging Face model:

```text
google/flan-t5-small
```

The model is loaded using the Hugging Face Transformers pipeline with:

```python
pipeline("text2text-generation")
```

The retrieved context is provided to the model so that the generated response can make use of the information retrieved from the knowledge base.

## Baseline vs RAG

One of the main objectives of this project is to compare two approaches.

### Baseline Generation

The language model receives only the question.

```text
Question
    |
    v
FLAN-T5
    |
    v
Baseline Answer
```

### RAG Generation

The language model receives both the retrieved context and the question.

```text
Question
    |
    v
FAISS Retrieval
    |
    v
Relevant Documents
    |
    v
Context + Question
    |
    v
FLAN-T5
    |
    v
RAG Answer
```

This comparison demonstrates how external retrieved information can be incorporated into the generation process.

## Example Query

The notebook uses the following query:

```text
How does AI improve interaction between humans and PCs?
```

The system:

1. Converts the query into an embedding.
2. Searches the FAISS index.
3. Retrieves the top three relevant documents.
4. Combines the retrieved documents into context.
5. Generates a baseline response without context.
6. Generates a RAG response using the retrieved context.
7. Displays both responses for comparison.

## Technologies Used

| Technology                | Purpose                                     |
| ------------------------- | ------------------------------------------- |
| Python                    | Core programming language                   |
| Sentence Transformers     | Generate document and query embeddings      |
| FAISS                     | Vector storage and similarity search        |
| Hugging Face Transformers | Text generation                             |
| FLAN-T5 Small             | Generative language model                   |
| PyTorch                   | Deep learning framework                     |
| NumPy                     | Numerical operations                        |
| Jupyter Notebook          | Development and experimentation environment |

## Library Versions

The notebook specifies the following versions:

```text
NumPy: 1.26.4
Accelerate: 1.7.0
Transformers: 4.52.4
Sentence Transformers: 4.1.0
FAISS CPU
```

## Installation

Install the required dependencies using:

```bash
pip install numpy==1.26.4
pip install accelerate==1.7.0
pip install transformers==4.52.4
pip install sentence-transformers==4.1.0
pip install faiss-cpu
```

Alternatively:

```bash
pip install numpy==1.26.4 accelerate==1.7.0 transformers==4.52.4 sentence-transformers==4.1.0 faiss-cpu
```

## How to Run

### 1. Clone the Repository

```bash
git clone <your-repository-link>
cd <repository-name>
```

### 2. Install Dependencies

```bash
pip install numpy==1.26.4 accelerate==1.7.0 transformers==4.52.4 sentence-transformers==4.1.0 faiss-cpu
```

### 3. Open the Notebook

Open the notebook using Jupyter Notebook or JupyterLab:

```bash
jupyter notebook
```

You can also open the notebook using Google Colab.

### 4. Run the Notebook

Execute the cells sequentially.

The notebook will:

* Load the required libraries
* Create the document knowledge base
* Generate document embeddings
* Build the FAISS index
* Convert the query into an embedding
* Retrieve the top three documents
* Generate a baseline response
* Generate a RAG response
* Compare both responses

## Project Structure

```text
Simple-RAG-FAISS/
|
├── faiss.ipynb
└── README.md
```

## Key Concepts Demonstrated

### Retrieval-Augmented Generation

RAG combines information retrieval with language generation. Instead of relying only on the information available within the language model, relevant external context is retrieved and provided to the model.

### Embeddings

Embeddings represent text as numerical vectors. Semantically related text can have similar vector representations.

### Vector Similarity Search

FAISS searches the embedding space to identify documents that are most similar to the query.

### L2 Distance

The FAISS index uses L2 distance to compare the query vector with document vectors.

### Context-Aware Generation

The retrieved documents are added to the prompt before sending the query to the generative model.

## Learning Objectives

This project helps build an understanding of:

* Retrieval-Augmented Generation
* Text embeddings
* Sentence Transformers
* Vector databases and vector search
* FAISS indexing
* Similarity search
* Query embeddings
* Context construction
* Prompt-based generation
* Hugging Face Transformers
* Comparing baseline and retrieval-augmented generation

## Limitations

This implementation is intentionally simple and uses a small in-memory document collection.

It does not include:

* Large-scale document ingestion
* Document chunking
* Persistent vector storage
* Metadata filtering
* Reranking
* Conversation memory
* Production deployment
* Advanced retrieval strategies

These components can be added when extending the project into a more complete RAG application.

## Possible Improvements

The pipeline can be extended by:

* Using a larger knowledge base
* Adding PDF and document ingestion
* Implementing document chunking
* Using persistent vector storage
* Experimenting with different embedding models
* Testing different language models
* Adding reranking
* Improving prompt templates
* Evaluating retrieval quality
* Building a user interface for interactive queries

## Conclusion

This project provides a simple and practical introduction to Retrieval-Augmented Generation.

By combining Sentence Transformers for embeddings, FAISS for similarity search, and FLAN-T5 for text generation, the pipeline demonstrates how retrieved information can be incorporated into a language model's prompt.

The baseline and RAG comparison provides a clear way to understand the role of retrieval in context-aware text generation.

## Author

Aayush Kumar

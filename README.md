# GitHub Repository Search and Chat System Report

## Methodology

### 1. Data Acquisition
Clone GitHub repositories using the `clone_repo` function.

### 2. Document Extraction and Loading
Traverse and load various file types from the repository using `load_and_index_files`.

### 3. Document Splitting and Embedding
Split documents into chunks with `RecursiveCharacterTextSplitter`.

### 4. Embedding Generation
Generate embeddings using SentenceTransformer ('all-MiniLM-L6-v2') and store them in Chroma.

### 5. Query Processing and Retrieval
Convert user queries to embeddings and retrieve relevant document chunks from Chroma using similarity search.

### 6. Response Generation
Create a detailed prompt with conversation history, query, and documents. Generate responses using the Ollama model (Llama2).

## Dataset Source
The dataset comprises various files from GitHub repositories, including code files, README files, and notebooks. These files are cloned directly from specified repositories, enabling the processing and indexing of diverse document types for effective information retrieval.

## Retrieval-Augmented Generation (RAG) Technique
A two-step RAG technique is leveraged:
1. Document retrieval after creating embeddings using SentenceTransformer and creating a vector database store.
2. Retrieved documents are provided to the Ollama model using Llama2 – 7B LLM. The prompt is especially curated for this purpose. Langchain was utilized for document loading, text splitting, and the RAG pipeline.

### Why?
1. **Using Langchain** simplifies the integration of several components and allows easy customization of the pipeline.
2. **Using Ollama for LLM** allows running Llama without GPU and produces contextually accurate answers.
3. **Sentence Transformer** provides high-quality embeddings.
4. **RecursiveCharacterTextSplitter** helps to split documents into manageable chunks.
5. **Curating a prompt** for the task utilizing several techniques improves the quality of answers.

## Vector Database
Chroma has been chosen as our vector database as it handles the embeddings in an efficient manner, providing semantic searching. It can quickly process and retrieve documents from large datasets, thereby increasing the accuracy of our application.

## Preventing Model Hallucinations
The following steps have been taken to prevent hallucinations:
1. The model generates answers strictly based on the retrieved documents and handles cases when it doesn’t have adequate context. The prompt has been curated to handle this.
2. A detailed context is provided, including the history and relevant documents and metadata, to limit hallucinations.
3. The prompt is curated to instruct the model to work in a step-by-step manner, providing reasoning to the context provided.
4. When asked about a particular file, the presence of that file in context is ensured by matching the query with metadata.

## Automated Checking for Response Accuracy
Semantic similarity scores can be used between the retrieved document and the model response to check whether the generated response is relevant to the context or not. Keyword matching can be done by identifying key entities in the answer and the retrieved document.

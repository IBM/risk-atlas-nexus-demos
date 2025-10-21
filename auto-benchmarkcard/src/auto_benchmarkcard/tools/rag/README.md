# RAG Tools for Benchmark Metadata

Enhanced retrieval-augmented generation system for fact verification in benchmark metadata. Combines multiple search strategies and LLM-based filtering to improve retrieval quality.

## Components

### RAGRetriever (`rag_retriever.py`)
Main retrieval system with configurable enhancement features.

```python
retriever = RAGRetriever(
    embedding_model="bge-large",      # or "e5-large", "minilm"
    enable_llm_reranking=True,        # Quality filtering with LLM
    enable_hybrid_search=True,        # Vector + BM25 search
    enable_query_expansion=True,      # Query reformulation
    llm_engine_type="RITS"           # or "OLLAMA"
)

retriever.index_documents(documents)
results = retriever.retrieve_for_statement("The dataset contains 70,000 examples")
```

### MetadataIndexer (`indexer.py`)
Converts JSON metadata into clean, searchable documents.

```python
indexer = MetadataIndexer()
documents = indexer.create_documents(unitxt_data, hf_data, benchmark_name, docling_data)
```

### BenchmarkCardAtomizer (`atomizer.py`)
Breaks benchmark cards into atomic factual statements for verification.

```python
statements = atomize_benchmark_card(benchmark_card, engine_type="rits")
# Returns: [{"text": "The dataset contains 70,000 examples", "field": "data.size"}, ...]
```

### Format Converter (`format_converter.py`)
Converts RAG results to required output format for fact verification pipeline.

## Architecture

1. **Document Processing**: Clean text extraction from JSON metadata
2. **Indexing**: Hierarchical chunking with vector and BM25 indices
3. **Atomization**: LLM breaks benchmark cards into verifiable facts
4. **Retrieval**: Multi-strategy search with quality filtering
5. **Output**: Structured results for fact verification

## Search Strategies

### Vector Search
- BGE-large embeddings (1024 dimensions)
- MMR for diversity
- Hierarchical chunks with parent text retrieval

### BM25 Search
- Keyword-based matching
- Complements vector search for exact term matching
- Custom implementation with tunable parameters

### Keyword Filtering
- Extracts important terms from queries
- Filters documents containing relevant keywords
- Catches proper nouns, numbers, technical terms

### LLM Reranking
- Scores chunks 1-10 for relevance
- Filters out headers and boilerplate
- Returns parent chunks for better context

## Configuration

### Embedding Models
- `bge-large`: BAAI/bge-large-en-v1.5 (recommended)
- `e5-large`: intfloat/e5-large-v2
- `minilm`: sentence-transformers/all-MiniLM-L6-v2 (fallback)

### LLM Integration
Requires `llm_handler` module for:
- Query reformulation
- Result reranking
- Benchmark card atomization

## Usage Example

```python
from tools.rag.rag_retriever import RAGRetriever
from tools.rag.indexer import MetadataIndexer
from tools.rag.atomizer import atomize_benchmark_card

# 1. Index documents
indexer = MetadataIndexer()
documents = indexer.create_documents(unitxt_data, hf_data, "hellaswag")

# 2. Setup enhanced retriever
retriever = RAGRetriever(
    persist_directory="./chroma_db",
    embedding_model="bge-large",
    enable_llm_reranking=True,
    enable_hybrid_search=True
)
retriever.index_documents(documents)

# 3. Atomize benchmark card
statements = atomize_benchmark_card(benchmark_card)

# 4. Retrieve evidence for each statement
for statement in statements:
    evidence = retriever.retrieve_for_statement(statement["text"])
    print(f"Statement: {statement['text']}")
    print(f"Evidence: {len(evidence)} chunks found")
```


## Dependencies

- `langchain-community`
- `langchain-core`
- `sentence-transformers`
- `chromadb`
- `langgraph`
- Custom `llm_handler` module for LLM integration

## Directory Structure

```
tools/rag/
├── README.md              # This file
├── rag_retriever.py       # Main retrieval system
├── indexer.py            # Document indexing
├── atomizer.py           # Statement atomization
├── format_converter.py   # Output formatting
├── output/              # RAG results output
└── chroma_db/           # Vector database storage
```

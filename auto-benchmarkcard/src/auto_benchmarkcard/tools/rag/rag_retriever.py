"""RAG retrieval system with enhanced search capabilities.

Builds on standard vector search with hierarchical chunking, hybrid search,
and LLM-based reranking for better fact verification.
"""

import asyncio
import json
import logging
import math
import re

# Suppress noisy logging from external libraries
import warnings
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, TypedDict

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# Suppress LangChain deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
warnings.filterwarnings("ignore", message=".*LangChainDeprecationWarning.*")
warnings.filterwarnings("ignore", message=".*HuggingFaceEmbeddings.*deprecated.*")
warnings.filterwarnings("ignore", message=".*manual persistence.*")

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    """State for the retrieval graph.

    Attributes:
        question: The query question.
        documents: List of retrieved documents.
        generation: Generated response text.
    """

    question: str
    documents: List[Document]
    generation: str


class RAGRetriever:
    """RAG retriever with enhanced search for fact verification.

    Combines vector search with BM25, uses hierarchical chunking to preserve
    context, and filters results with LLM reranking to improve quality.

    Args:
        persist_directory: Optional path to store vector database (None for in-memory)
        embedding_model: "bge-large", "e5-large", or "minilm"
        enable_llm_reranking: Use LLM to score and filter results
        enable_hybrid_search: Combine vector search with BM25
        enable_query_expansion: Reformulate queries with LLM
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        embedding_model: str = "bge-large",
        enable_llm_reranking: bool = True,
        enable_hybrid_search: bool = True,
        enable_query_expansion: bool = True,
    ):

        self.embedding_model = embedding_model
        self.embeddings = self._initialize_embeddings(embedding_model)

        self.persist_directory = persist_directory
        self.vectorstore = None
        self.retriever = None
        self.enable_llm_reranking = enable_llm_reranking
        self.enable_hybrid_search = enable_hybrid_search
        self.enable_query_expansion = enable_query_expansion

        # BM25 components for hybrid search
        self.bm25_index = None
        self.documents_for_bm25 = []

        # LLM handler for query expansion and reranking - use default from config
        self.llm_handler = None
        if enable_llm_reranking or enable_query_expansion:
            try:
                from auto_benchmarkcard.config import get_llm_handler
                self.llm_handler = get_llm_handler()
                logger.debug("Using default LLM handler from config")
            except Exception as e:
                logger.warning(f"Failed to get LLM handler from config: {e}")
                logger.warning("Disabling LLM features")
                self.enable_llm_reranking = False
                self.enable_query_expansion = False

        self.app = self._build_graph()

        logger.debug("RAG retriever initialized")

    def _initialize_embeddings(self, embedding_model: str) -> HuggingFaceEmbeddings:
        """Initialize embedding model based on choice.

        Args:
            embedding_model: Model name ("bge-large", "e5-large", or "minilm")

        Returns:
            Configured HuggingFaceEmbeddings instance
        """
        if embedding_model == "bge-large":
            return HuggingFaceEmbeddings(
                model_name="BAAI/bge-large-en-v1.5",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        elif embedding_model == "e5-large":
            return HuggingFaceEmbeddings(
                model_name="intfloat/e5-large-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        else:  # minilm fallback
            return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def _chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents using hierarchical chunking.

        Uses chunk sizes from configuration. Overlap is automatically set to ~10% of chunk size.

        Args:
            documents: List of documents to chunk.

        Returns:
            List of chunked documents with parent-child relationships in metadata.
        """
        # Get chunk sizes from configuration
        from auto_benchmarkcard.config import Config
        parent_size = Config.PARENT_CHUNK_SIZE
        child_size = Config.CHILD_CHUNK_SIZE

        # Calculate overlaps (~10% of chunk size)
        parent_overlap = int(parent_size * 0.1)
        child_overlap = int(child_size * 0.125)  # Slightly higher for small chunks

        # Parent chunks: large context windows
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size, chunk_overlap=parent_overlap, separators=["\n\n", "\n", ". ", " ", ""]
        )

        # Child chunks: precise retrieval
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size, chunk_overlap=child_overlap, separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunked_docs: List[Document] = []

        for doc in documents:
            # Create parent chunks
            parent_chunks = parent_splitter.split_text(doc.page_content)

            for parent_idx, parent_text in enumerate(parent_chunks):
                parent_id = f"{id(doc)}_parent_{parent_idx}"

                # Create child chunks from each parent
                child_chunks = child_splitter.split_text(parent_text)

                for child_idx, child_text in enumerate(child_chunks):
                    child_meta = dict(doc.metadata) | {
                        "parent_id": parent_id,
                        "parent_text": parent_text,  # Store parent for retrieval
                        "chunk_index": child_idx,
                        "is_child_chunk": True,
                    }
                    chunked_docs.append(Document(page_content=child_text, metadata=child_meta))

        logger.debug(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
        return chunked_docs

    def _build_bm25_index(self, documents: List[Document]) -> None:
        """Build BM25 index for keyword-based search.

        Args:
            documents: List of Document objects to index
        """
        if not self.enable_hybrid_search:
            return

        try:
            self.documents_for_bm25 = documents

            # Tokenize documents
            tokenized_docs = []
            for doc in documents:
                tokens = re.findall(r"\b\w+\b", doc.page_content.lower())
                tokenized_docs.append(tokens)

            # Build simple BM25 index
            self.bm25_index = {
                "tokenized_docs": tokenized_docs,
                "doc_freqs": defaultdict(int),
                "idf": {},
                "doc_lens": [len(doc) for doc in tokenized_docs],
                "avgdl": (
                    sum(len(doc) for doc in tokenized_docs) / len(tokenized_docs)
                    if tokenized_docs
                    else 0
                ),
            }

            # Calculate document frequencies
            for tokens in tokenized_docs:
                unique_tokens = set(tokens)
                for token in unique_tokens:
                    self.bm25_index["doc_freqs"][token] += 1

            # Calculate IDF values
            num_docs = len(tokenized_docs)
            for token, df in self.bm25_index["doc_freqs"].items():
                self.bm25_index["idf"][token] = math.log((num_docs - df + 0.5) / (df + 0.5))

            logger.debug(f"BM25 index built for {num_docs} documents")

        except Exception as e:
            logger.warning(f"Failed to build BM25 index: {e}")
            self.enable_hybrid_search = False

    def _bm25_search(self, query: str, k: int = 10) -> List[tuple]:
        """Search using BM25 algorithm.

        Args:
            query: Search query string.
            k: Number of top results to return.

        Returns:
            List of (document_index, score) tuples sorted by relevance.
        """
        if not self.enable_hybrid_search or not self.bm25_index:
            return []

        try:
            # Tokenize query
            query_tokens = re.findall(r"\b\w+\b", query.lower())

            # BM25 parameters
            k1, b = 1.5, 0.75

            scores = []
            for doc_idx, tokens in enumerate(self.bm25_index["tokenized_docs"]):
                score = 0
                token_freqs = Counter(tokens)
                doc_len = self.bm25_index["doc_lens"][doc_idx]

                for query_token in query_tokens:
                    if query_token in self.bm25_index["idf"]:
                        tf = token_freqs[query_token]
                        idf = self.bm25_index["idf"][query_token]

                        # BM25 formula
                        numerator = tf * (k1 + 1)
                        denominator = tf + k1 * (1 - b + b * (doc_len / self.bm25_index["avgdl"]))
                        score += idf * (numerator / denominator)

                scores.append((doc_idx, score))

            # Sort by score and return top k
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:k]

        except Exception as e:
            logger.warning(f"BM25 search failed: {e}")
            return []

    def index_documents(self, documents: List[Document]) -> None:
        """Index documents into the vector store.

        Applies hierarchical chunking and builds both vector and BM25 indices.

        Args:
            documents: List of documents to index.
        """
        documents = self._chunk_documents(documents)

        try:
            if self.vectorstore is None:
                self.vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                )
            else:
                self.vectorstore.add_documents(documents)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to index documents: {error_msg}")
            raise

        self._build_bm25_index(documents)

        # Configure retriever with MMR for diversity
        from auto_benchmarkcard.config import Config
        # If LLM reranking enabled, fetch more candidates for the LLM to filter
        # Otherwise just fetch top-k directly
        k_value = Config.DEFAULT_TOP_K * 3 if self.enable_llm_reranking else Config.DEFAULT_TOP_K
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k_value,
                "fetch_k": k_value * 2 + 5,  # Fetch more for MMR diversity
                "lambda_mult": 0.8,
            },
        )

    def _build_graph(self):
        """Build the retrieval graph.

        Returns:
            Compiled StateGraph for document retrieval workflow.
        """
        workflow = StateGraph(GraphState)

        # Define nodes
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)

        # Build graph with optional LLM reranking
        workflow.add_edge(START, "retrieve")
        if self.enable_llm_reranking:
            workflow.add_node("llm_rerank", self.llm_rerank_documents)
            workflow.add_edge("retrieve", "llm_rerank")
            workflow.add_edge("llm_rerank", "grade_documents")
        else:
            workflow.add_edge("retrieve", "grade_documents")
        workflow.add_edge("grade_documents", END)

        return workflow.compile()

    def retrieve(self, state: GraphState) -> Dict[str, Any]:
        """Basic document retrieval step.

        Args:
            state: Current graph state with question.

        Returns:
            Updated state with retrieved documents.
        """
        question = state["question"]
        documents = self.retriever.invoke(question)
        return {"documents": documents}

    def grade_documents(self, state: GraphState) -> Dict[str, Any]:
        """Filter out short and duplicate documents.

        Args:
            state: Current graph state with documents.

        Returns:
            Updated state with filtered documents.
        """
        documents = state["documents"]
        filtered_docs = []
        seen_content = set()

        for doc in documents:
            content = doc.page_content.strip()
            if len(content) < 20:
                continue

            # Simple deduplication
            is_duplicate = any(content in seen or seen in content for seen in seen_content)
            if not is_duplicate:
                filtered_docs.append(doc)
                seen_content.add(content)

        return {"documents": filtered_docs}

    def llm_rerank_documents(self, state: GraphState) -> Dict[str, Any]:
        """Use LLM to score and filter documents by relevance.

        Args:
            state: Current graph state with documents and question.

        Returns:
            Updated state with reranked and filtered documents.
        """
        documents = state["documents"]
        question = state["question"]

        if not documents or not self.enable_llm_reranking or not self.llm_handler:
            return {"documents": documents}

        try:
            # Prepare chunks for scoring
            chunks_text = "\n\n".join(
                [
                    f"Chunk {i+1}:\n{doc.page_content[:500]}..."
                    for i, doc in enumerate(documents[:10])
                ]
            )

            rerank_prompt = f"""Query: "{question}"

Score each chunk 1-10 for relevance. Filter out headers and boilerplate.

Chunks:
{chunks_text}

Return JSON array: [8, 3, 9, 1, 7, 2, 6, 4, 5, 3]"""

            response = self.llm_handler.generate(rerank_prompt)
            scores = self._parse_scores(response)

            if not scores:
                return {"documents": documents[:3]}

            # Keep high-scoring chunks and return parent text when available
            scored_docs = list(zip(documents[: len(scores)], scores))
            filtered_docs = []
            for doc, score in scored_docs:
                # Ensure score is numeric before comparison
                if isinstance(score, (int, float)) and score >= 6:
                    filtered_docs.append(doc)

            # Sort by score (highest first)
            filtered_docs.sort(
                key=lambda doc: next(
                    (
                        score
                        for d, score in scored_docs
                        if d == doc and isinstance(score, (int, float))
                    ),
                    0,
                ),
                reverse=True,
            )

            final_docs = []
            for doc in filtered_docs[:3]:
                if doc.metadata.get("parent_text"):
                    parent_doc = Document(
                        page_content=doc.metadata["parent_text"], metadata=doc.metadata
                    )
                    final_docs.append(parent_doc)
                else:
                    final_docs.append(doc)

            logger.debug(f"Reranked {len(documents)} → {len(final_docs)} documents")
            return {"documents": final_docs}

        except Exception as e:
            logger.error(f"LLM reranking failed: {e}")
            return {"documents": documents[:3]}

    def _parse_scores(self, response: str) -> List[int]:
        """Parse LLM response to extract relevance scores.

        Args:
            response: LLM response containing JSON array of scores.

        Returns:
            List of integer scores, or empty list if parsing fails.
        """
        try:
            response = response.strip()
            start_idx = response.find("[")
            end_idx = response.rfind("]") + 1
            if start_idx >= 0 and end_idx > start_idx:
                parsed_scores = json.loads(response[start_idx:end_idx])
                # Ensure all scores are integers
                return [
                    int(score) if isinstance(score, (int, float)) else 0 for score in parsed_scores
                ]
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.debug(f"Failed to parse scores: {e}")
        return []

    async def _async_rerank_documents(
        self, statement: str, documents: List[Document]
    ) -> List[Document]:
        """Async version of document reranking for parallel processing.

        Args:
            statement: Statement to find relevant documents for.
            documents: List of candidate documents.

        Returns:
            List of top reranked documents.
        """
        if not documents or not self.enable_llm_reranking or not self.llm_handler:
            return documents[:3]

        try:
            # Prepare chunks for scoring
            chunks_text = "\n\n".join(
                [
                    f"Chunk {i+1}:\n{doc.page_content[:500]}..."
                    for i, doc in enumerate(documents[:10])
                ]
            )

            rerank_prompt = f"""Query: "{statement}"

Score each chunk 1-10 for relevance. Filter out headers and boilerplate.

Chunks:
{chunks_text}

Return JSON array: [8, 3, 9, 1, 7, 2, 6, 4, 5, 3]"""

            # Run LLM call in thread pool to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self.llm_handler.generate, rerank_prompt)
            scores = self._parse_scores(response)

            if not scores:
                return documents[:3]

            # Keep high-scoring chunks and return parent text when available
            scored_docs = list(zip(documents[: len(scores)], scores))
            filtered_docs = []
            for doc, score in scored_docs:
                if isinstance(score, (int, float)) and score >= 6:
                    filtered_docs.append(doc)

            # Sort by score (highest first)
            filtered_docs.sort(
                key=lambda doc: next(
                    (
                        score
                        for d, score in scored_docs
                        if d == doc and isinstance(score, (int, float))
                    ),
                    0,
                ),
                reverse=True,
            )

            final_docs = []
            for doc in filtered_docs[:3]:
                if doc.metadata.get("parent_text"):
                    parent_doc = Document(
                        page_content=doc.metadata["parent_text"], metadata=doc.metadata
                    )
                    final_docs.append(parent_doc)
                else:
                    final_docs.append(doc)

            logger.debug(f"Async reranked {len(documents)} → {len(final_docs)} documents")
            return final_docs

        except Exception as e:
            logger.error(f"Async LLM reranking failed: {e}")
            return documents[:3]

    def extract_keywords(self, statement: str) -> List[str]:
        """Extract important terms from statement for keyword search.

        Args:
            statement: Statement to extract keywords from.

        Returns:
            List of extracted keyword strings.
        """
        keywords = []

        # Clean up statement
        cleaned = re.sub(r"^(The|A|An)\s+", "", statement, flags=re.IGNORECASE)
        cleaned = re.sub(r"(is|are|was|were|has|have)\s+", " ", cleaned, flags=re.IGNORECASE)

        # Extract different types of terms
        patterns = [
            (r'"([^"]+)"', lambda m: m.group(1)),  # Quoted strings
            (r"https?://[^\s]+", lambda m: m.group(0)),  # URLs
            (
                r"\b\d+[,\d]*(?:\.\d+)?\s*(?:k|K|thousand|million|%|examples?|MB|GB)?\b",
                lambda m: m.group(0),
            ),  # Numbers
            (
                r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b",
                lambda m: m.group(0),
            ),  # Proper nouns
            (
                r"\b(?:benchmark|dataset|accuracy|license|MIT|JSON|API|NLP|AI|BERT|model)\b",
                lambda m: m.group(0).lower(),
            ),  # Tech terms
        ]

        for pattern, extractor in patterns:
            matches = re.finditer(pattern, cleaned, flags=re.IGNORECASE)
            keywords.extend([extractor(m) for m in matches])

        # Clean and deduplicate
        keywords = list(set([kw.strip() for kw in keywords if len(kw.strip()) > 2]))
        logger.debug(f"Extracted {len(keywords)} keywords from statement")
        return keywords

    def keyword_filter_documents(
        self, keywords: List[str], candidate_pool_size: int = 20
    ) -> List[Document]:
        """Filter documents containing any of the keywords.

        Args:
            keywords: List of keywords to search for.
            candidate_pool_size: Maximum number of documents to return.

        Returns:
            List of documents matching the keywords.
        """
        if not keywords:
            return []

        try:
            broad_query = " ".join(keywords[:3])
            all_docs = self.vectorstore.similarity_search(broad_query, k=candidate_pool_size * 2)
        except Exception:
            logger.warning("Could not retrieve documents for keyword filtering")
            return []

        filtered_docs = []
        for doc in all_docs:
            content_lower = doc.page_content.lower()
            if any(keyword.lower() in content_lower for keyword in keywords):
                filtered_docs.append(doc)

        logger.debug(f"Keyword filtering found {len(filtered_docs)} matching documents")
        return filtered_docs[:candidate_pool_size]

    def _reformulate_atoms_for_search_batch(self, statements: List[str]) -> List[str]:
        """Reformulate multiple atomic statements into better search queries in one LLM call.

        Args:
            statements: List of atomic statements to reformulate.

        Returns:
            List of reformulated search queries in the same order as input.
        """
        if not self.llm_handler or not statements:
            return statements

        try:
            # Create numbered list of statements
            numbered_statements = []
            for i, statement in enumerate(statements, 1):
                numbered_statements.append(f"{i}. {statement}")

            statements_text = "\n".join(numbered_statements)

            prompt = f"""Turn these factual statements into search queries to find evidence:

{statements_text}

Focus on key terms, numbers, and specific entities. Remove generic words.
Return as JSON array: ["query1", "query2", "query3", ...]"""

            response = self.llm_handler.generate(prompt).strip()

            # Parse the JSON response
            try:
                start_idx = response.find("[")
                end_idx = response.rfind("]") + 1
                if start_idx >= 0 and end_idx > start_idx:
                    reformulated_queries = json.loads(response[start_idx:end_idx])

                    # Sanity check and fallback
                    if isinstance(reformulated_queries, list) and len(reformulated_queries) == len(
                        statements
                    ):

                        validated_queries = []
                        for i, query in enumerate(reformulated_queries):
                            if isinstance(query, str) and len(query.strip()) >= 3:
                                validated_queries.append(query.strip())
                            else:
                                # Fallback to original statement
                                validated_queries.append(statements[i])

                        logger.debug(f"Batch reformulated {len(statements)} queries")
                        return validated_queries
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.debug(f"Failed to parse batch reformulation response: {e}")

            # Fallback to original statements
            return statements

        except Exception as e:
            logger.warning(f"Batch query reformulation failed: {e}")
            return statements

    def _reformulate_atom_for_search(self, statement: str) -> str:
        """Reformulate single atomic statement into better search query (fallback method).

        Args:
            statement: Single statement to reformulate.

        Returns:
            Reformulated search query string.
        """
        batch_result = self._reformulate_atoms_for_search_batch([statement])
        return batch_result[0] if batch_result else statement

    def retrieve_for_statement(self, statement: str) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a factual statement.

        Uses hybrid search combining vector similarity, BM25, and keyword matching,
        with optional LLM reranking for quality filtering.
        """
        if not self.retriever:
            raise ValueError("No documents indexed yet!")

        # Reformulate statement into better search query
        search_query = self._reformulate_atom_for_search(statement)

        # Show which atom we're processing
        atom_preview = statement[:80] + "..." if len(statement) > 80 else statement
        logger.debug(f'🔄 Retrieving evidence for: "{atom_preview}"')

        # Extract keywords from both original and reformulated queries
        keywords = self.extract_keywords(statement)
        reformulated_keywords = self.extract_keywords(search_query)
        all_keywords = list(set(keywords + reformulated_keywords))

        # Collect candidates from multiple search methods
        all_candidates = []
        seen_content = set()

        # Vector search
        try:
            vector_results = self.retriever.invoke(search_query)
            for doc in vector_results:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_content:
                    all_candidates.append(doc)
                    seen_content.add(content_hash)
            logger.debug(f"Vector search found {len(vector_results)} documents")
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            vector_results = []

        # BM25 search (if enabled)
        if self.enable_hybrid_search and self.bm25_index:
            try:
                bm25_results = self._bm25_search(search_query, k=15)
                bm25_docs = [
                    self.documents_for_bm25[idx] for idx, score in bm25_results if score > 0
                ]
                for doc in bm25_docs:
                    content_hash = hash(doc.page_content)
                    if content_hash not in seen_content:
                        all_candidates.append(doc)
                        seen_content.add(content_hash)
                logger.debug(f"BM25 search found {len(bm25_docs)} additional documents")
            except Exception as e:
                logger.warning(f"BM25 search failed: {e}")

        # Keyword filtering
        keyword_filtered = self.keyword_filter_documents(all_keywords, candidate_pool_size=10)
        for doc in keyword_filtered:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                all_candidates.append(doc)
                seen_content.add(content_hash)

        # Use hybrid results if we have enough, otherwise fall back to vector
        final_candidates = all_candidates[:20]
        if len(final_candidates) >= 5:
            documents_to_process = final_candidates
        else:
            logger.debug("Using vector search fallback")
            documents_to_process = vector_results

        # Process through graph (includes reranking if enabled)
        result = self.app.invoke({"question": statement, "documents": documents_to_process})

        # Format results
        retrieved_chunks = []
        for doc in result["documents"]:
            retrieved_chunks.append(
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "source": doc.metadata.get("source", "unknown"),
                    "type": doc.metadata.get("type", "unknown"),
                }
            )

        # Show summary of what we found
        sources = set(chunk.get("source", "unknown") for chunk in retrieved_chunks)
        logger.debug(f"✅ Found {len(retrieved_chunks)} evidence chunks from {len(sources)} sources")
        return retrieved_chunks

    def retrieve_for_statements_batch(self, statements: List[str]) -> List[List[Dict[str, Any]]]:
        """Retrieve relevant documents for multiple factual statements using batch query reformulation.

        More efficient than individual calls as it reformulates all queries in one LLM call.

        Args:
            statements: List of factual statements to find evidence for

        Returns:
            List of retrieved chunks for each statement (same order as input)
        """
        if not self.retriever:
            raise ValueError("No documents indexed yet!")

        if not statements:
            return []

        logger.debug(f"🔄 Processing {len(statements)} statements with batch reformulation")

        # Batch reformulate all statements at once (single LLM call)
        reformulated_queries = self._reformulate_atoms_for_search_batch(statements)

        # Now process each statement with its reformulated query
        all_results = []
        for i, (statement, search_query) in enumerate(zip(statements, reformulated_queries)):
            # Show which atom we're processing
            atom_preview = statement[:80] + "..." if len(statement) > 80 else statement
            logger.debug(f'🔄 [{i+1}/{len(statements)}] Processing: "{atom_preview}"')

            # Extract keywords from both original and reformulated queries
            keywords = self.extract_keywords(statement)
            reformulated_keywords = self.extract_keywords(search_query)
            all_keywords = list(set(keywords + reformulated_keywords))

            # Collect candidates from multiple search methods
            all_candidates = []
            seen_content = set()

            # Vector search
            try:
                vector_results = self.retriever.invoke(search_query)
                for doc in vector_results:
                    content_hash = hash(doc.page_content)
                    if content_hash not in seen_content:
                        all_candidates.append(doc)
                        seen_content.add(content_hash)
                logger.debug(f"Vector search found {len(vector_results)} documents")
            except Exception as e:
                logger.warning(f"Vector search failed: {e}")
                vector_results = []

            # BM25 search (if enabled)
            if self.enable_hybrid_search and self.bm25_index:
                try:
                    bm25_results = self._bm25_search(search_query, k=15)
                    bm25_docs = [
                        self.documents_for_bm25[idx] for idx, score in bm25_results if score > 0
                    ]
                    for doc in bm25_docs:
                        content_hash = hash(doc.page_content)
                        if content_hash not in seen_content:
                            all_candidates.append(doc)
                            seen_content.add(content_hash)
                    logger.debug(f"BM25 search found {len(bm25_docs)} additional documents")
                except Exception as e:
                    logger.warning(f"BM25 search failed: {e}")

            # Keyword filtering
            keyword_filtered = self.keyword_filter_documents(all_keywords, candidate_pool_size=10)
            for doc in keyword_filtered:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_content:
                    all_candidates.append(doc)
                    seen_content.add(content_hash)

            # Use hybrid results if we have enough, otherwise fall back to vector
            final_candidates = all_candidates[:20]
            if len(final_candidates) >= 5:
                documents_to_process = final_candidates
            else:
                logger.debug("Using vector search fallback")
                documents_to_process = vector_results

            # Process through graph (includes reranking if enabled)
            result = self.app.invoke({"question": statement, "documents": documents_to_process})

            # Format results
            retrieved_chunks = []
            for doc in result["documents"]:
                retrieved_chunks.append(
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "source": doc.metadata.get("source", "unknown"),
                        "type": doc.metadata.get("type", "unknown"),
                    }
                )

            all_results.append(retrieved_chunks)

            # Show summary for this statement
            sources = set(chunk.get("source", "unknown") for chunk in retrieved_chunks)
            logger.debug(f"Found {len(retrieved_chunks)} chunks from {len(sources)} sources")

        # Show final summary
        total_chunks = sum(len(result) for result in all_results)
        logger.debug(f"✅ Batch processing complete: {total_chunks} total chunks retrieved")

        return all_results

    async def retrieve_for_statements_batch_parallel(
        self, statements: List[str]
    ) -> List[List[Dict[str, Any]]]:
        """Async version with parallel reranking for improved performance.

        Same as retrieve_for_statements_batch but uses async parallel reranking
        when LLM reranking is enabled, providing 3-5x speedup.

        Args:
            statements: List of factual statements to find evidence for

        Returns:
            List of retrieved chunks for each statement (same order as input)
        """
        if not self.retriever:
            raise ValueError("No documents indexed yet!")

        if not statements:
            return []

        logger.debug(f"🚀 Processing {len(statements)} statements with parallel reranking")

        # Batch reformulate all statements at once (single LLM call)
        reformulated_queries = self._reformulate_atoms_for_search_batch(statements)

        # Collect all document candidates for each statement (sequential part)
        statements_and_docs = []
        for i, (statement, search_query) in enumerate(zip(statements, reformulated_queries)):
            atom_preview = statement[:80] + "..." if len(statement) > 80 else statement
            logger.debug(f'🔄 [{i+1}/{len(statements)}] Collecting docs for: "{atom_preview}"')

            # Extract keywords from both original and reformulated queries
            keywords = self.extract_keywords(statement)
            reformulated_keywords = self.extract_keywords(search_query)
            all_keywords = list(set(keywords + reformulated_keywords))

            # Collect candidates from multiple search methods
            all_candidates = []
            seen_content = set()

            # Vector search
            try:
                vector_results = self.retriever.invoke(search_query)
                for doc in vector_results:
                    content_hash = hash(doc.page_content)
                    if content_hash not in seen_content:
                        all_candidates.append(doc)
                        seen_content.add(content_hash)
                logger.debug(f"Vector search found {len(vector_results)} documents")
            except Exception as e:
                logger.warning(f"Vector search failed: {e}")
                vector_results = []

            # BM25 search (if enabled)
            if self.enable_hybrid_search and self.bm25_index:
                try:
                    bm25_results = self._bm25_search(search_query, k=15)
                    bm25_docs = [
                        self.documents_for_bm25[idx] for idx, score in bm25_results if score > 0
                    ]
                    for doc in bm25_docs:
                        content_hash = hash(doc.page_content)
                        if content_hash not in seen_content:
                            all_candidates.append(doc)
                            seen_content.add(content_hash)
                    logger.debug(f"BM25 search found {len(bm25_docs)} additional documents")
                except Exception as e:
                    logger.warning(f"BM25 search failed: {e}")

            # Keyword filtering
            keyword_filtered = self.keyword_filter_documents(all_keywords, candidate_pool_size=10)
            for doc in keyword_filtered:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_content:
                    all_candidates.append(doc)
                    seen_content.add(content_hash)

            # Use hybrid results if we have enough, otherwise fall back to vector
            final_candidates = all_candidates[:20]
            if len(final_candidates) >= 5:
                documents_to_process = final_candidates
            else:
                logger.debug("Using vector search fallback")
                documents_to_process = vector_results

            statements_and_docs.append((statement, documents_to_process))

        # Now do parallel reranking if enabled
        if self.enable_llm_reranking and self.llm_handler:
            logger.debug(f"Starting parallel LLM reranking for {len(statements)} statements")

            # Create async tasks for parallel reranking
            rerank_tasks = [
                self._async_rerank_documents(statement, documents)
                for statement, documents in statements_and_docs
            ]

            # Execute all reranking tasks concurrently
            reranked_docs_list = await asyncio.gather(*rerank_tasks)
        else:
            # No reranking - just apply basic filtering
            reranked_docs_list = []
            for statement, documents in statements_and_docs:
                # Apply basic grade_documents filtering
                filtered_docs = []
                seen_content = set()
                for doc in documents:
                    content = doc.page_content.strip()
                    if len(content) < 20:
                        continue
                    # Simple deduplication
                    is_duplicate = any(content in seen or seen in content for seen in seen_content)
                    if not is_duplicate:
                        filtered_docs.append(doc)
                        seen_content.add(content)
                reranked_docs_list.append(filtered_docs[:3])

        # Format final results
        all_results = []
        for i, (statement, reranked_docs) in enumerate(zip(statements, reranked_docs_list)):
            retrieved_chunks = []
            for doc in reranked_docs:
                retrieved_chunks.append(
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "source": doc.metadata.get("source", "unknown"),
                        "type": doc.metadata.get("type", "unknown"),
                    }
                )

            all_results.append(retrieved_chunks)

            # Show summary for this statement
            sources = set(chunk.get("source", "unknown") for chunk in retrieved_chunks)
            logger.debug(
                f"Statement {i+1}: {len(retrieved_chunks)} chunks from {len(sources)} sources"
            )

        # Show final summary
        total_chunks = sum(len(result) for result in all_results)
        logger.debug(f"Parallel batch processing complete: {total_chunks} total chunks retrieved")

        return all_results

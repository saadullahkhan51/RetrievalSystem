from .config import Config
from .database import Database
from .embedding import Embedder
from .indexing import FAISSIndex
from .vector_counter import VectorCounter
import os
from typing import Dict, List
import nltk
nltk.download('punkt')
import numpy as np

class EmbeddingRetrievalSystem:
    def __init__(self, config: Config):
        self.config = config
        self.db = Database(self.config.db_path)
        self.embedder = Embedder(self.config.model_name)
        self.index = FAISSIndex(self.embedder.model.get_sentence_embedding_dimension())
        self.vector_counter = VectorCounter(self.db)

    async def add_documents(self, directory: str):
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                file_path = os.path.join(directory, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    await self._process_document(filename, content)

    async def _process_document(self, filename: str, content: str):
        doc_id = self.db.add_document(filename, content)
        chunks = self._split_into_chunks(content)
        
        for i in range(0, len(chunks), self.config.batch_size):
            batch = chunks[i:i+self.config.batch_size]
            embeddings = await self._generate_embeddings(batch)
            
            for chunk, embedding in zip(batch, embeddings):
                self.db.add_chunk(doc_id, chunk, embedding.tobytes())
                self.index.add(embedding.reshape(1, -1))

    def _split_into_chunks(self, text: str) -> List[str]:
        # Implementation of text splitting logic
        # This could be a simple split by sentences or a more complex method
        # For simplicity, let's split by sentences:
        sentences = text.split('. ')
        chunks = ['. '.join(sentences[i:i+self.config.chunk_size]) for i in range(0, len(sentences), self.config.chunk_size)]
        return chunks

    async def _generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        return self.embedder.encode(texts)

    async def search(self, query: str, top_k: int = 1) -> List[Dict[str, float]]:
        query_embedding = await self._generate_embeddings([query])
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            chunk = self.db.get_chunk(int(idx) + 1)
            if chunk:
                results.append({
                    "score": float(score),
                    "chunk": chunk['content']
                })
        
        return results

    async def generate_response(self, query: str) -> str:
        results = await self.search(query)
        if not results:
            return "I couldn't find any relevant information for your query."
        
        relevant_chunk = results[0]['chunk']
        response = f"Based on the query '{query}', here's what I found:\n\n{relevant_chunk}\n\n"
        response += f"This information has a relevance score of {results[0]['score']:.2f}."
        
        return response

    def close(self):
        self.db.close()
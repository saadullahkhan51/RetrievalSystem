from .config import Config
from .database import Database
from .embedding import Embedder
from .indexing import FAISSIndex
import os
from typing import Dict, List
import nltk
nltk.download('punkt')
import numpy as np
import json

class EmbeddingRetrievalSystem:
    def __init__(self, config: Config):
        self.config = config
        self.db = Database(self.config.db_path)
        self.embedder = Embedder(self.config.model_name)
        self.index = FAISSIndex(self.embedder.model.get_sentence_embedding_dimension(), self.config.index_path)
        self.processed_files = self._load_processed_files()

    async def add_documents(self, directory: str):
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                file_path = os.path.join(directory, filename)
                file_stat = os.stat(file_path)
                file_mtime = file_stat.st_mtime

                if filename in self.processed_files and self.processed_files[filename] == file_mtime:
                    print(f"Skipping {filename} as it hasn't changed since last processing.")
                    continue

                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    await self._process_document(filename, content)
                
                self.processed_files[filename] = file_mtime
        
        self._save_processed_files()

    def _load_processed_files(self):
        processed_files_path = os.path.join(os.path.dirname(self.config.processed_files_path), "processed_files.json")
        if os.path.exists(processed_files_path):
            with open(processed_files_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_processed_files(self):
        processed_files_path = os.path.join(os.path.dirname(self.config.processed_files_path), "processed_files.json")
        with open(processed_files_path, 'w') as f:
            json.dump(self.processed_files, f)

    async def _process_document(self, filename: str, content: str):
        doc_id = self.db.add_document(filename, content)
        chunks = self._split_into_chunks(content)
        
        for i in range(0, len(chunks), self.config.batch_size):
            batch = chunks[i:i+self.config.batch_size]
            embeddings = await self._generate_embeddings(batch)
            
            for chunk, embedding in zip(batch, embeddings):
                self.db.add_chunk(doc_id, chunk, embedding.tobytes())
                self.index.add(embedding.reshape(1, -1))
        self.index.save()

    def _split_into_chunks(self, text: str) -> List[str]:
        # Implementation of text splitting logic
        # This could be a simple split by sentences or a more complex method
        # For simplicity, split by sentences:
        sentences = text.split('. ')
        chunks = ['. '.join(sentences[i:i+self.config.chunk_size]) for i in range(0, len(sentences), self.config.chunk_size)]
        return chunks

    async def _generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        return self.embedder.encode(texts)
    
    async def rebuild_index(self):
        self.index = FAISSIndex(self.embedder.model.get_sentence_embedding_dimension(), self.config.index_path)
        chunks_with_embeddings = self.db.get_all_chunks_with_embeddings()

        # Process embeddings in batches
        batch_size = self.config.batch_size
        for i in range(0, len(chunks_with_embeddings), batch_size):
            batch = chunks_with_embeddings[i:i+batch_size]
            embeddings = [np.frombuffer(chunk['embedding'], dtype=np.float32) for chunk in batch]
            embeddings_array = np.vstack(embeddings)
            self.index.add(embeddings_array)

        self.index.save()
        print(f"Index rebuilt with {len(chunks_with_embeddings)} embeddings.")

    async def search(self, query: str, top_k: int = 3) -> List[Dict[str, float]]:
        query = self._preprocess_query(query)
        query_embedding = await self._generate_embeddings([query])
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for i in range(top_k):
            chunk = self.db.get_chunk(int(indices[0][i]) + 1)
            if chunk:
                results.append({
                    "score": float(scores[0][i]),
                    "chunk": chunk['content']
                })
        return results
    
    def _preprocess_query(self, query: str) -> str:
        # Remove punctuation, lowercase, etc.
        return query.lower().strip()
    
    async def generate_response(self, query: str, k: int = 3) -> str:
        results = await self.search(query, k)
        if not results:
            return "I couldn't find any relevant information for your query."
        
        response = f"Query: '{query}'\n\nTop {k} relevant chunks:\n\n"
        for i, result in enumerate(results, 1):
            response += f"{i}. Score: {result['score']:.4f}\n   Chunk: {result['chunk']}\n\n"

        return response

    def close(self):
        self.index.save()
        self.db.close()
        self._save_processed_files()
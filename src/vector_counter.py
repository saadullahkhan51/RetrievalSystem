import sqlite3
from typing import Tuple

class VectorCounter:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def count_vectors(self) -> Tuple[int, int]:
        """
        Count the number of documents and chunks (vectors) in the database.
        
        Returns:
        Tuple[int, int]: (number of documents, number of chunks/vectors)
        """
        self.cursor.execute("SELECT COUNT(*) FROM documents")
        doc_count = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT COUNT(*) FROM chunks")
        chunk_count = self.cursor.fetchone()[0]
        
        return doc_count, chunk_count

    def estimate_vector_count(self, sample_size: int = 100) -> float:
        """
        Estimate the total number of vectors based on a sample of documents.
        
        Args:
        sample_size (int): Number of documents to sample for estimation.
        
        Returns:
        float: Estimated total number of vectors.
        """
        self.cursor.execute(f"SELECT id FROM documents ORDER BY RANDOM() LIMIT {sample_size}")
        sample_ids = [row[0] for row in self.cursor.fetchall()]
        
        placeholders = ','.join('?' for _ in sample_ids)
        self.cursor.execute(f"SELECT COUNT(*) FROM chunks WHERE document_id IN ({placeholders})", sample_ids)
        sample_chunk_count = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT COUNT(*) FROM documents")
        total_doc_count = self.cursor.fetchone()[0]
        
        estimated_total_chunks = (sample_chunk_count / len(sample_ids)) * total_doc_count
        
        return estimated_total_chunks

    def close(self):
        self.conn.close()
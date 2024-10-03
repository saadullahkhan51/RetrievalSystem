import sqlite3
from typing import List, Dict

class Database:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def _create_tables(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                filename TEXT,
                content TEXT
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                document_id INTEGER,
                content TEXT,
                embedding BLOB,
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
        ''')
        self.conn.commit()

    def add_document(self, filename: str, content: str) -> int:
        self.cursor.execute(
            'INSERT INTO documents (filename, content) VALUES (?, ?)',
            (filename, content)
        )
        self.conn.commit()
        return self.cursor.lastrowid

    def add_chunk(self, document_id: int, content: str, embedding: bytes):
        self.cursor.execute(
            'INSERT INTO chunks (document_id, content, embedding) VALUES (?, ?, ?)',
            (document_id, content, embedding)
        )
        self.conn.commit()

    def document_exists(self, filename: str) -> bool:
        query = "SELECT COUNT(*) FROM documents WHERE filename = ?"
        self.cursor.execute(query, (filename,))
        return self.cursor.fetchone()[0] > 0

    def get_document(self, document_id: int) -> Dict[str, str]:
        self.cursor.execute('SELECT filename, content FROM documents WHERE id = ?', (document_id,))
        result = self.cursor.fetchone()
        return {'filename': result[0], 'content': result[1]} if result else None
    
    def get_all_documents(self) -> Dict[str, str]:
        self.cursor.execute('SELECT id, filename FROM documents')
        return [{'id': row[0], 'filename': row[1]} for row in self.cursor.fetchall()]

    def get_chunk(self, chunk_id: int) -> Dict[str, str]:
        self.cursor.execute('SELECT content FROM chunks WHERE id = ?', (chunk_id,))
        result = self.cursor.fetchone()
        return {'content': result[0]} if result else None

    def get_all_chunks(self) -> List[Dict[str, str]]:
        self.cursor.execute('SELECT id, content FROM chunks')
        return [{'id': row[0], 'content': row[1]} for row in self.cursor.fetchall()]
    
    def get_all_chunks_with_embeddings(self) -> List[Dict]:
        query = "SELECT id, content, embedding FROM chunks"
        self.cursor.execute(query)
        return [{"id": row[0], "content": row[1], "embedding": row[2]} for row in self.cursor.fetchall()]

    def close(self):
        self.conn.close()
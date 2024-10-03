import unittest
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch
from src.config import Config
from src.retrieval_system import EmbeddingRetrievalSystem

class TestEmbeddingRetrievalSystem(unittest.TestCase):
    def setUp(self):
        self.config = Mock(spec=Config)
        self.config.db_path = ':memory:'
        self.config.model_name = 'test-model'
        self.config.batch_size = 2
        self.config.chunk_size = 2

        self.retrieval_system = EmbeddingRetrievalSystem(self.config)

    def tearDown(self):
        self.retrieval_system.close()

    @patch('src.embedding.Embedder.encode')
    def test_generate_embeddings(self, mock_encode):
        mock_encode.return_value = [[1.0, 2.0], [3.0, 4.0]]
        
        texts = ['Hello world', 'Test sentence']
        result = asyncio.run(self.retrieval_system._generate_embeddings(texts))
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].tolist(), [1.0, 2.0])
        self.assertEqual(result[1].tolist(), [3.0, 4.0])

    def test_split_into_chunks(self):
        text = "This is a test. It has multiple sentences. We want to split it."
        chunks = self.retrieval_system._split_into_chunks(text)
        
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0], "This is a test. It has multiple sentences.")
        self.assertEqual(chunks[1], "We want to split it.")

    @patch('src.embedding.Embedder.encode')
    @patch('src.indexing.FAISSIndex.add')
    def test_process_document(self, mock_index_add, mock_encode):
        mock_encode.return_value = [[1.0, 2.0], [3.0, 4.0]]
        
        filename = "test.txt"
        content = "This is a test. It has multiple sentences. We want to process it."
        
        asyncio.run(self.retrieval_system._process_document(filename, content))
        
        # Check if document was added to the database
        self.assertTrue(self.retrieval_system.db.document_exists(filename))
        
        # Check if chunks were added to the database and index
        self.assertEqual(mock_index_add.call_count, 2)

    @patch('src.embedding.Embedder.encode')
    @patch('src.indexing.FAISSIndex.search')
    def test_search(self, mock_search, mock_encode):
        mock_encode.return_value = [[1.0, 2.0]]
        mock_search.return_value = ([0.9, 0.8], [[0, 1]])
        
        # Add some test chunks to the database
        self.retrieval_system.db.add_document("test.txt", "Test content")
        self.retrieval_system.db.add_chunk(1, "Chunk 1", b'test_embedding')
        self.retrieval_system.db.add_chunk(1, "Chunk 2", b'test_embedding')
        
        query = "Test query"
        results = asyncio.run(self.retrieval_system.search(query))
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['score'], 0.9)
        self.assertEqual(results[0]['chunk'], "Chunk 1")
        self.assertEqual(results[1]['score'], 0.8)
        self.assertEqual(results[1]['chunk'], "Chunk 2")

    @patch('src.retrieval_system.EmbeddingRetrievalSystem.search')
    def test_generate_response(self, mock_search):
        mock_search.return_value = [
            {"score": 0.9, "chunk": "This is a relevant chunk."}
        ]
        
        query = "Test query"
        response = asyncio.run(self.retrieval_system.generate_response(query))
        
        self.assertIn("Test query", response)
        self.assertIn("This is a relevant chunk.", response)
        self.assertIn("0.90", response)

    def test_add_documents(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some test documents
            with open(os.path.join(temp_dir, "doc1.txt"), "w") as f:
                f.write("This is document 1.")
            with open(os.path.join(temp_dir, "doc2.txt"), "w") as f:
                f.write("This is document 2.")
            
            # Add documents
            asyncio.run(self.retrieval_system.add_documents(temp_dir))
            
            # Check if documents were added
            self.assertTrue(self.retrieval_system.db.document_exists("doc1.txt"))
            self.assertTrue(self.retrieval_system.db.document_exists("doc2.txt"))

    def test_rebuild_index(self):
        self.retrieval_system.rebuild_index()

if __name__ == '__main__':
    unittest.main()
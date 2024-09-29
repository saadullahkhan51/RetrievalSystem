import faiss
import os

class FAISSIndex:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)

    def add(self, vectors):
        self.index.add(vectors)

    def search(self, query_vector, k):
        return self.index.search(query_vector, k)
    
    def serialize_faiss_index(index, file_path):
        """
        Serialize a FAISS index to disk.
        
        Args:
        index (faiss.Index): The FAISS index to serialize.
        file_path (str): Path where the index will be saved.
        """
        faiss.write_index(index, file_path)
        print(f"Index serialized to {file_path}")

    def deserialize_faiss_index(file_path): 
        """
        Deserialize a FAISS index from disk.
        
        Args:
        file_path (str): Path where the index is saved.
        
        Returns:
        faiss.Index: The deserialized FAISS index.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No index file found at {file_path}")
        
        index = faiss.read_index(file_path)
        print(f"Index deserialized from {file_path}")
        return index
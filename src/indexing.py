import faiss
import os

class FAISSIndex:
    def __init__(self, dim: int, index_path: str = None):
        if index_path and os.path.exists(index_path):
            self.index = self._deserialize_faiss_index(index_path)
        else:
            self.index = faiss.IndexFlatIP(dim)
        self.index_path = index_path

    def add(self, vectors):
        self.index.add(vectors)

    def search(self, query_vector, k):
        return self.index.search(query_vector, k)
    
    def _serialize_faiss_index(self):
        """
        Serialize the FAISS index to disk.
        """
        if self.index_path:
            faiss.write_index(self.index, self.index_path)
            print(f"Index serialized to {self.index_path}")
        else:
            raise ValueError("No index_path specified for serialization")

    def _deserialize_faiss_index(self, file_path): 
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

    def save(self, index_path: str = None):
        """
        Save the current index to disk.
        """
        if index_path: self.index_path = index_path
        self._serialize_faiss_index()

    def load(self, file_path: str = None):
        """
        Load an index from disk.
        
        Args:
        file_path (str, optional): Path to the index file. If not provided, uses the instance's index_path.
        """
        path = file_path or self.index_path
        if path:
            self.index = self._deserialize_faiss_index(path)
            if not file_path:
                self.index_path = path
        else:
            raise ValueError("No file path provided for loading the index")
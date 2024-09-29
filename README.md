To use this `EmbeddingRetrievalSystem`:

Initialize the system with a config file:
```py
pythonCopyretrieval_system = EmbeddingRetrievalSystem('path/to/config.yaml')
```

Add documents from a directory:
```py
pythonCopyawait retrieval_system.add_documents('path/to/document/directory')
```

Generate a response to a query:
```py
pythonCopyresponse = await retrieval_system.generate_response("What is machine learning?")
print(response)
```

Close the system when done:
```py
pythonCopyretrieval_system.close()
```
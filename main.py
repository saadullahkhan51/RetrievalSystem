import asyncio
from src.config import Config
from src.retrieval_system import EmbeddingRetrievalSystem

async def main():
    config_path = "config/config.yaml"
    config = Config(config_path)
    retrieval_system = EmbeddingRetrievalSystem(config)
 
    documents_directory = config.documents_path
    await retrieval_system.add_documents(documents_directory)

    query = "What is the relationship between AI and Machine Learning?"
    response = await retrieval_system.generate_response(query)
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
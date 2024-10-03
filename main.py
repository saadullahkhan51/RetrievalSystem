import asyncio
from src.config import Config
from src.retrieval_system import EmbeddingRetrievalSystem

async def query_loop(retrieval_system: EmbeddingRetrievalSystem):
    print("\nEntering query mode. Type 'exit' to quit.")
    while True:
        query = input("\nEnter your query: ")
        if query.lower() == 'exit':
            break
        try:
            response = await retrieval_system.generate_response(query)
            print("\nResponse:")
            print(response.Result())
        except Exception as e:
            print(f"Error processing query: {str(e)}")

async def main():
    config_path = "config/config.yaml"
    config = Config(config_path)
    retrieval_system = EmbeddingRetrievalSystem(config)
 
    documents_directory = config.documents_path
    await retrieval_system.add_documents(documents_directory)

    await query_loop(retrieval_system)

    retrieval_system.close() 

if __name__ == "__main__":
    asyncio.run(main())
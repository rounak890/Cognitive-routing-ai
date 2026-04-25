
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain_community.docstore.document import Document


# USING GEMINI FOR EMBEDDING MODEL
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv() # to get the gemini API key from .env file

# making execution logs using loguru
from loguru import logger
logger.add('logs/router_execution.log')


# 1. Initialize embeddings
# embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
# gemini embeddings were giving issues, so witched to hugging face embeddings and they worked perfectly
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # no API key required



# 2. Bot personas - kept same as were described in assignment doc
personas = [
    ("bot_a", "I believe AI and crypto will solve all human problems. I am highly optimistic about technology, Elon Musk, and space exploration."),
    ("bot_b", "I believe late-stage capitalism and tech monopolies are destroying society. I am highly critical of AI, social media, and billionaires."),
    ("bot_c", "I strictly care about markets, interest rates, trading algorithms, and making money.")
]

# making doc of the personas to make the vector db 
docs = [
    Document(page_content=text, metadata={"bot_id": bot_id}) for bot_id, text in personas
]

logger.info("Generated documents for vector store:")
logger.info(docs)


# 3. Create vector DB from earlier made docs
vectorstore = FAISS.from_documents(docs, embeddings)

# 4. Router main function
def route_post_to_bots(post_content: str, threshold: float = 0.1) -> list:
    results = vectorstore.similarity_search_with_score(post_content, k=3)

    matched = []
    for doc, score in results:
        similarity = 1 / (1 + score)  
        if similarity > threshold:
            matched.append(doc.metadata["bot_id"])

    return matched


# checking if the function works properly
if __name__ == "__main__":
    post = "OpenAI released a new AI model replacing developers"
    threshold = 0.4

    logger.info(f"routing post: {post}")
    logger.info(f"current threhold : {threshold}")
    logger.info(f"matched bots: {route_post_to_bots(post, threshold)}")

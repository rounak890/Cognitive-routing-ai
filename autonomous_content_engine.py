from langchain.tools import tool
from dotenv import load_dotenv
import json

load_dotenv() # to get the gemini API key from .env file


from loguru import logger
logger.add("logs/content_engine_execution.logs")


# making the tool as described in the assignment for mock results - note that providing docstring to the fucntion is imp
@tool
def mock_searxng_search(query: str) -> str:
    """
    Python function which provides recent news headlines based on keywords in the query. 
    For example, if the query contains "crypto", it returns "Bitcoin hits new all-time high amid regulatory ETF approvals". 
    If the query contains "AI", it returns "New AI model disrupts software industry". 
    For any other query, it returns "Global markets remain uncertain amid inflation fears".

    Args:
        query (str) - The search query containing keywords.
    
    """
    if "crypto" in query.lower():
        return "Bitcoin hits new all-time high after ETF approval"
    if "ai" in query.lower():
        return "New AI model disrupts software industry"
    
    return "Global markets remain uncertain amid inflation fears"


# rest code of graph making and tool calling
from langgraph.graph import StateGraph
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview", #gemini-3.1-pro-preview
)


class GraphState(TypedDict):
    bot_id: str
    persona: str
    topic: str
    search_results: str
    post_content: str

# node 1: decide topic
def decide_topic(state):
    logger.info(f"Starting node 1 - decide_topic for persona: {state['persona']}")
    prompt = f"""
    You are this persona:
    {state['persona']}

    Decide what topic you want to post today.
    Return ONLY the topic name, nothing else.
    """
    topic = llm.invoke(prompt).content
    logger.info(f"Decided topic: {topic}")
    return {"topic": topic[0].get('text', '').strip("'")}  # remove extra quotes if any

# node 2: search

def search(state):
    logger.info(f"Starting node 2 - search")
    logger.info(f"Searching for topic: {state['topic']}")
    result = mock_searxng_search.run(state["topic"])
    logger.info(f"Search result: {result}")
    return {"search_results": result}

# node 3: draft
def draft(state):
    logger.info(f"Starting node 3 - draft for topic: {state['topic']}")
    prompt = f"""
    Persona: {state['persona']}
    Topic: {state['topic']}
    Context: {state['search_results']}

    Write a 280 character opinionated post.

    Output JSON:
    {{
        "bot_id": "{state['bot_id']}",
        "topic": "...",
        "post_content": "..."
    }}
    """
    response = llm.invoke(prompt).content
    response_text = response[0].get('text', '')
    # parse JSON string response
    json_data = json.loads(response_text.strip("```json\n").strip("````"))
    logger.info(f"Draft completed with bot_id: {json_data.get('bot_id')}, topic: {json_data.get('topic')}")
    return {"post_content": json_data}


# graph making 
builder = StateGraph(GraphState)
builder.add_node("decide", decide_topic)
builder.add_node("search", search)
builder.add_node("draft", draft)

builder.set_entry_point("decide")
builder.add_edge("decide", "search")
builder.add_edge("search", "draft")

graph = builder.compile()

# TEST
if __name__ == "__main__":
    logger.info("Starting autonomous content engine execution")

    logger.info(f"""Input data ->  "bot_id": "bot_a", "persona": "Tech maximalist who loves AI and crypto" """)
    result = graph.invoke({
        "bot_id": "bot_a",
        "persona": "Tech maximalist who loves AI and crypto",
        "topic": None,
        "search_results": None,
        "post_content": None
    })
    logger.info(f"Final post content: {result['post_content']}")



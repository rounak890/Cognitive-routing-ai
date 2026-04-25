Assignment by Rounak Gera

Running Steps:-
1. add your gemini api key in .env file
2. install the dependencies
3. run the files


# Briefly explain your LangGraph node structure 

I have intentionally kept the langgraph structure simple as it was something sequential and simple

the structrue have 3 nodes
1. decide_topic : calls llm to get the topic name
2. search : get resent information from our mock tool
3. draft : calls llm to get the final reply/output along with parsing it to json 

all these are sequential


STRUCTURE :- decide_topic -> search -> draft

--------------------------------------------------------

# how you chose to defend against the prompt injection in Phase 3.

in this i prevent by following ways:-
- identifying if the user prompt is malicious(by lookign up for several keywords)

- accordingly i draft a system prompt to prevent from prompt injection(the prompt contains IDENTITY LOCK, IDENTITY LOCK, IDENTITY LOCK and ACTIVE INJECTION ALERT)

- i also modify the user prompt and put a [SECURITY ALERT] if the prompt seems malicious


note :- it is not a production ready solution to prevent from prompt injection attacks as such attacks grow rapidly and a good solution would be to use another small llm as an evaluator and active user tracking to have user history of attempting attacks. 
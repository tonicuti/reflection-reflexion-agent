import datetime

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

from schemas import AnswerQuestion, ReviseAnswer

llm = ChatOllama(
    model = "qwen3:1.7b",
)

structured_llm_answer_question = llm.with_structured_output(
    AnswerQuestion,
    method="json_schema",
)

structured_llm_revise_answer = llm.with_structured_output(
    ReviseAnswer,
    method="json_schema",
)

# Responder
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an expert researcher.
            Current time: {time}
            
            1. {first_instruction}
            2. Reflect and critique your answer. Be severe to maximize the improvement.
            3. Recommand search queries to research information and improve your answer. 
            """
        ),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Answer the user's question above using the required format."
        )
    ]
).partial(
    time = lambda: datetime.datetime.now().isoformat()
)

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction = "Provide a detailed ~250 words answer"
)

first_responder = first_responder_prompt_template | structured_llm_answer_question


# Revisor
revise_instructions = """ 
    Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
    - You MUST include numerical citations in your revised answer to ensure it can be verified.
    - Add a "References" section at the end of your answer with the sources you used to improve your answer. In form of:
        - [1] https://www.example.com
        - [2] https://www.example.com
    - You should use the previous critique to remove superfluous information from your answer and make sure it is not more than 250 words.
"""

revisor = actor_prompt_template.partial(
    first_instruction = revise_instructions
) | structured_llm_revise_answer

if __name__ == "__main__":
    human_message = HumanMessage(
        content = "Write about AI-Powered SOC / Autonomous SOC problem domain,"
                    " list startups that do that and raised capital."
    )
    
    res = first_responder.invoke({"messages": [human_message]})

    print(res)
    print("Search queries:", res.search_queries)

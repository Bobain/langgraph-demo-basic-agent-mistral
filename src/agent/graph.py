"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

# To support Python < 3.12 which is used in LangGraph Docker image with langgraph up
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
import langsmith as ls  # noqa: F401
from pydantic import BaseModel
from langchain_tavily import TavilySearch

MAX_TAVILY_RESULTS = 5
TAVILY_INJECT_RAW_USER_MEASSAGE = False


## This Sate allows additional fields to be added on top of the of ones already defined,
#   therefore : it would allow any criterie "mentionned" by the user but not in the list
# @dataclass
# class Criteres:
#     criteres: dict[str, bool | None] = field(
#         default_factory=lambda: {
#             "plage": None,
#             "montagne": None,
#             "ville": None,
#             "sport": None,
#             "detente": None,
#             "acces_handicap": None,
#         }
#     )


# This version is more strict and will ignore extra fields
class Criteres(BaseModel):
    plage: bool | None = None
    montagne: bool | None = None
    ville: bool | None = None
    sport: bool | None = None
    detente: bool | None = None
    acces_handicap: bool | None = None


ai_role_message = f"""Bonjour, je suis un assistant qui va vous aider à planifier votre prochain voyage.
        Vous pouvez me parler naturellement, mais sachez que je vais limiter mes recherches de voyages aux critères suivants:
        {', '.join(list(Criteres.model_fields.keys()))}
        """


class Context(TypedDict):
    """Context parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    my_configurable_param: str


@dataclass
class State:
    """Input state for the agent.

    Defines the initial structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """

    last_user_message: str
    ai_structured_output: Criteres | None = None
    last_ai_message: str = ""
    message_count: int = 0


async def criteria_no_match(state: State):
    return {
        "last_user_message": "",
        "message_count": state.message_count + 1,
        "ai_structured_output": None,
        "last_ai_message": ai_role_message,
        # f"Configured with {runtime.context.get('my_configurable_param')}"
    }


async def criteria_router(state: State):
    if any(v is not None for v in state.ai_structured_output.model_dump().values()):
        return search_travel.__name__
    else:
        return criteria_no_match.__name__


async def criteria_extractor_model(state: State) -> Dict[str, Any]:
    """Process input and returns output.

    , runtime: Runtime[Context]
    Cannot use runtime context to alter behavior.
    """
    print(str(state))
    # see https://docs.mistral.ai/getting-started/models/models_overview/

    chat_model = init_chat_model(model="codestral-2508", model_provider="mistralai")
    chat_model = chat_model.with_structured_output(Criteres)

    res = await chat_model.ainvoke(state.last_user_message)
    return {
        "last_user_message": state.last_user_message,
        "message_count": state.message_count + 1,
        "ai_structured_output": res,
        "last_ai_message": state.last_ai_message,
        # f"Configured with {runtime.context.get('my_configurable_param')}"
    }


async def search_travel(state: State):
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    wrapped = TavilySearch(max_results=MAX_TAVILY_RESULTS)
    criteria = state.ai_structured_output
    query = "Trouve un voyage tout organisé qui respecte les critères suivants \n: "
    for criterion, value in criteria.model_dump().items():
        if value is not None and not criterion == "acces_handicap":
            query += f"{criterion} = {value} "

    if criteria.acces_handicap:
        query += (
            "\n ATTENTION cette personne est handicapée, "
            "il faut absolument que tu trouves un voyage qui respecte les critères mais soit accessible à une personne handicapée"
        )
    res = await wrapped.ainvoke({"query": query})

    results = []
    for result in res["results"]:
        results.append(result["title"] + " : " + result["url"])

    print(results)

    return {
        "last_user_message": state.last_user_message,
        "message_count": state.message_count + 1,
        "ai_structured_output": state.ai_structured_output,
        "last_ai_message": "\n".join(results),
    }


# Define the graph
builder = StateGraph(State, context_schema=Context)
builder.add_node(criteria_extractor_model.__name__, criteria_extractor_model)
builder.add_node(criteria_no_match.__name__, criteria_no_match)
builder.add_node(search_travel.__name__, search_travel)

builder.add_edge(START, criteria_extractor_model.__name__)

builder.add_conditional_edges(criteria_extractor_model.__name__, criteria_router)
builder.add_edge(search_travel.__name__, END)
builder.add_edge(criteria_no_match.__name__, END)

graph = builder.compile(name="Travel searcher")


if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv

    load_dotenv()

    print(graph.get_graph().draw_ascii())

    # state = State(last_user_message="J'aime le sport et la randonnée dans le désert")
    state = State(last_user_message="J'aime le sport et la randonnée, mais je suis une personne à mobilité réduite")
    state = State(last_user_message="J'aime la randonnée, mais je suis PMR")
    state = State(last_user_message="J'aime la randonnée à kla campagne, mais je suis PMR")
    state = State(last_user_message="J'aime les balades à à dos de dromadaire. Je suis PMR")
    # state = State(last_user_message="Bonjour")
    # state = State(last_user_message="J'aime le sport et la et la randonnée")
    print(asyncio.run(graph.ainvoke(state)))

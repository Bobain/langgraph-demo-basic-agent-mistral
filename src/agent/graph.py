"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

# To support Python < 3.12 which is used in LangGraph Docker image with langgraph up
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph
import langsmith as ls  # noqa: F401
from pydantic import BaseModel

## This structure output allows additional fields to be added like randonnee
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


async def call_model(state: State) -> Dict[str, Any]:
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
        "message_count": state.message_count + 1,
        "ai_structured_output": res,
        # f"Configured with {runtime.context.get('my_configurable_param')}"
    }


# Define the graph
builder = StateGraph(State, context_schema=Context).add_node(call_model).add_edge("__start__", "call_model")

graph = builder.compile(name="New Graph")


if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv

    load_dotenv()
    # state = State(last_user_message="J'aime le sport et la randonnée dans le désert")
    state = State(last_user_message="J'aime le sport et la randonnée, mais je suis une personne à mobilité réduite")
    # state = State(last_user_message="J'aime le sport et la et la randonnée")
    print(asyncio.run(graph.ainvoke(state)))

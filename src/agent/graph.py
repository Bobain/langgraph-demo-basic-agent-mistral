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


def model_introduction(state: State):
    return {
        "last_user_message": state.last_user_message,
        "message_count": state.message_count + 1,
        "ai_structured_output": None,
        "last_ai_message": ai_role_message,
        # f"Configured with {runtime.context.get('my_configurable_param')}"
    }


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
        "last_user_message": state.last_user_message,
        "message_count": state.message_count + 1,
        "ai_structured_output": res,
        "last_ai_message": ai_role_message,
        # f"Configured with {runtime.context.get('my_configurable_param')}"
    }


# Define the graph
builder = StateGraph(State, context_schema=Context)
builder.add_node(model_introduction.__name__, model_introduction)
builder.add_node(call_model.__name__, call_model)

builder.add_edge("__start__", "call_model")
builder.add_conditional_edges("model_introduction", "call_model")

graph = builder.compile(name="New Graph")


if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv

    load_dotenv()
    # state = State(last_user_message="J'aime le sport et la randonnée dans le désert")
    state = State(last_user_message="J'aime le sport et la randonnée, mais je suis une personne à mobilité réduite")
    # state = State(last_user_message="J'aime le sport et la et la randonnée")
    print(asyncio.run(graph.ainvoke(state)))


#
#
# criteres = Criteres()
#
# # Obtenir tous les champs avec leurs valeurs
# for field_name, value in criteres:
#     print(f"{field_name}: {value}")
#
# # Ou avec model_dump()
# criteres_dict = criteres.model_dump()
# for field_name, value in criteres_dict.items():
#     print(f"{field_name}: {value}")
#
# Pour votre cas d'usage dans le code :
#
# Voici comment vous pourriez l'utiliser dans votre fonction has_criteria ou search_travel :
#
# def has_criteria(state: State) -> str:
#     """Check if any criteria is True."""
#     criteres = state.ai_structured_output
#     if criteres is None:
#         return "ask_criteria"
#
#     # Obtenir toutes les valeurs des champs
#     values = criteres.model_dump().values()
#
#     # Vérifier si au moins une valeur est True
#     if any(values):
#         return "search"
#     else:
#         return "ask_criteria"
#
# Ou pour construire dynamiquement la liste des critères actifs :
#
# async def search_travel(state: State) -> Dict[str, Any]:
#     """Search for travel options using Tavily based on criteria."""
#     criteres = state.ai_structured_output
#
#     # Obtenir tous les critères actifs (True)
#     active_criteria = [
#         field_name
#         for field_name, value in criteres.model_dump().items()
#         if value is True
#     ]
#
#     query = f"voyage destination {' '.join(active_criteria)}"
#     # ...
#
# La méthode recommandée pour Pydantic v2 est model_fields pour accéder aux métadonnées des champs et model_dump() pour obtenir les valeurs d'une instance.
#

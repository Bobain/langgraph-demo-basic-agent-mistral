"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations
from dataclasses import dataclass
from pydantic import BaseModel
import unidecode
from typing_extensions import TypedDict
from random import choice
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
import langsmith as ls  # noqa: F401

OUTPUT_TRAVELS = [
    {"nom": "Randonnée camping en Lozère", "labels": ["sport", "montagne", "campagne"], "accessibleHandicap": "non"},
    {
        "nom": "5 étoiles à Chamonix option fondue",
        "labels": ["montagne", "détente"],
        "accessibleHandicap": "oui",
    },
    {
        "nom": "5 étoiles à Chamonix option ski",
        "labels": ["montagne", "sport"],
        "accessibleHandicap": "non",
    },
    {
        "nom": "Palavas de paillotes en paillotes",
        "labels": ["plage", "ville", "détente", "paillote"],
        "accessibleHandicap": "oui",
    },
    {
        "nom": "5 étoiles en rase campagne",
        "labels": ["campagne", "détente"],
        "accessibleHandicap": "oui",
    },
]


class Criteres(BaseModel):
    plage: bool | None = None
    montagne: bool | None = None
    ville: bool | None = None
    sport: bool | None = None
    detente: bool | None = None
    acces_handicap: bool | None = None


def match_criteria_and_travels(criteres: Criteres) -> str:
    if criteres.acces_handicap:
        output_travels = []
        for travels in OUTPUT_TRAVELS:
            if travels["accessibleHandicap"] == "oui":
                output_travels.append(travels)
    else:
        output_travels = OUTPUT_TRAVELS

    scores_for_travels = [0] * len(output_travels)

    for num_travel, travel in enumerate(output_travels):
        labels = [unidecode.unidecode(label) for label in travel["labels"]]
        for criterion, criterion_yes_no in criteres.model_dump().items():
            if criterion == "acces_handicap":
                continue
            if (not criterion_yes_no) and (criterion in labels):
                scores_for_travels[num_travel] += -1
            if criterion_yes_no and (criterion in labels):
                scores_for_travels[num_travel] += 1

    max_score = -10 * len(criteres.model_dump().keys())
    num_best_travel = None
    for i, score in enumerate(scores_for_travels):
        if score > max_score:
            max_score = score
            num_best_travel = i
        elif score == max_score:
            num_best_travel = choice([num_best_travel, i])

    return output_travels[num_best_travel]


class MessageUnderstandable(BaseModel):
    answer: bool | None = None


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


async def chat_model(state: State):
    """Process input and returns output.

    , runtime: Runtime[Context]
    Cannot use runtime context to alter behavior.
    """
    print(str(state))
    # see https://docs.mistral.ai/getting-started/models/models_overview/

    chat_model = init_chat_model(model="codestral-2508", model_provider="mistralai")
    res = await chat_model.with_structured_output(MessageUnderstandable).ainvoke(
        f"Le message suivant est-il compréhensible? \n {state.last_user_message}"
    )

    if not res.answer:
        return {
            "last_user_message": state.last_user_message,
            "message_count": state.message_count + 1,
            "ai_structured_output": state.ai_structured_output,
            "last_ai_message": "Désolé je n'ai pas compris votre message",
            # f"Configured with {runtime.context.get('my_configurable_param')}"
        }

    criteres = await chat_model.with_structured_output(Criteres).ainvoke(state.last_user_message)
    print(criteres)
    if all(v is None for v in res.model_dump().values()):
        return {
            "last_user_message": state.last_user_message,
            "message_count": state.message_count + 1,
            "ai_structured_output": None,
            "last_ai_message": f"""Bonjour, je suis un assistant qui va vous aider à planifier votre prochain voyage.
                Vous pouvez me parler naturellement, mais sachez que je vais limiter mes recherches de voyages aux critères suivants:
                {', '.join(list(Criteres.model_fields.keys()))}
                """,
            # f"Configured with {runtime.context.get('my_configurable_param')}"
        }

    if state.ai_structured_output is None:
        # Initiating Criteria, we set them to None as asked, when not mentionned
        for key, val in criteres.model_dump().items():
            if val is None:
                setattr(criteres, key, False)
        state.ai_structured_output = criteres
    else:
        # Updating Criteria
        #   we don't set to False a criterion which was previously mentioned but is not in the last message
        for key, val in criteres.model_dump().items():
            if val is not None:
                setattr(state.ai_structured_output, key, val)

    return {
        "last_user_message": state.last_user_message,
        "message_count": state.message_count + 1,
        "ai_structured_output": state.ai_structured_output,
        "last_ai_message": str(match_criteria_and_travels(state.ai_structured_output))
        + "\nVous pouvez préciser votre demande afin que je réponde mieux à vos attentes",
        # f"Configured with {runtime.context.get('my_configurable_param')}"
    }


# Define the graph
builder = StateGraph(State, context_schema=Context)

builder.add_node(chat_model.__name__, chat_model)

builder.add_edge(START, chat_model.__name__)
builder.add_edge(chat_model.__name__, END)

graph = builder.compile(name="Travel searcher")


if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv

    load_dotenv()

    state = State(last_user_message="Bonjour, j'aime la montagne")
    print(asyncio.run(graph.ainvoke(state)))

"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

# To support Python < 3.12 which is used in LangGraph Docker image with langgraph up
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
import langsmith as ls  # noqa: F401


@dataclass
class Criteres:
    criteres: dict[str, bool | None] = field(
        default_factory=lambda: {
            "plage": None,
            "montagne": None,
            "ville": None,
            "sport": None,
            "detente": None,
            "acces_handicap": None,
        }
    )


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
    ai_structured_output: Criteres
    # last_ai_message: str = ""
    # message_count: int = 0


async def call_model(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Process input and returns output.

    Can use runtime context to alter behavior.
    """
    print(str(state))
    # see https://docs.mistral.ai/getting-started/models/models_overview/

    chat_model = init_chat_model(model="codestral-2508", model_provider="mistralai")
    chat_model = chat_model.with_structured_output(Criteres)

    await chat_model.ainvoke(state.last_user_message)
    return {
        # "message_count": state.message_count + 1,
        # "last_ai_message": res.content,  # "output from call_model."
        "ai_structured_output": state.ai_structured_output,
        # f"Configured with {runtime.context.get('my_configurable_param')}"
    }


# Define the graph
builder = StateGraph(State, context_schema=Context).add_node(call_model).add_edge("__start__", "call_model")

graph = builder.compile(name="New Graph")

"""Message state, written out by hand (the same shape langgraph.graph.message ships).

MessagesState holds one `messages` list. The add_messages reducer appends what a node returns instead of replacing
the list. _format_messages converts messages to the OpenAI chat format and back, to normalise them.
"""
import warnings
from collections.abc import Sequence
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, BaseMessage, convert_to_messages
from langgraph.graph.message import add_messages


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def _format_messages(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    try:
        from langchain_core.messages import convert_to_openai_messages
    except ImportError:
        msg = (
            "Must have langchain-core>=0.3.11 installed to use automatic message "
            "formatting (format='langchain-openai'). Please update your langchain-core "
            "version or remove the 'format' flag. Returning un-formatted "
            "messages."
        )
        warnings.warn(msg)
        return list(messages)
    else:
        return convert_to_messages(convert_to_openai_messages(messages))

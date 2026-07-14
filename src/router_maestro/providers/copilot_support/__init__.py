"""Internal collaborators behind the stable :mod:`providers.copilot` facade."""

from router_maestro.providers.copilot_support.auth_session import CopilotAuthSession
from router_maestro.providers.copilot_support.catalog import CopilotCatalog
from router_maestro.providers.copilot_support.chat_codec import CopilotChatCodec
from router_maestro.providers.copilot_support.responses_codec import CopilotResponsesCodec
from router_maestro.providers.copilot_support.transport import CopilotTransport

__all__ = [
    "CopilotAuthSession",
    "CopilotCatalog",
    "CopilotChatCodec",
    "CopilotResponsesCodec",
    "CopilotTransport",
]

from .toolace_handle import ToolACEMultiTurnMessages
from .xlam_handle import XLAMMultiTurnMessages
from .xlam2_handle import XLAM2MultiTurnMessages
from .gorilla_handle import GorillaMultiTurnMessages
from .api_handle import APIMultiTurnMessages
from .llama_handle import LlamaMultiTurnMessages
from .chatglm_handle import ChatGLMMultiTurnMessages
from .hammer_handle import HammerMultiTurnMessages
from .watt_handle import WattMultiTurnMessages
from .fcm_handle import FCMMultiTurnMessages
from .hunyuan_handle import HunyuanMultiTurnMessages


tool_handle_map = {
    # hunyuan
    "hunyuan-turbos-latest": (APIMultiTurnMessages, False),
    "hunyuan-a13b": (HunyuanMultiTurnMessages, False),
    # toolace
    "toolace": (ToolACEMultiTurnMessages, False),
    "toolace2": (ToolACEMultiTurnMessages, False),
    # xlam
    "xlam": (XLAMMultiTurnMessages, False),
    "xlam2-70b": (XLAM2MultiTurnMessages, False),
    "xlam2-32b": (XLAM2MultiTurnMessages, False),
    "xlam2-8b": (XLAM2MultiTurnMessages, False),
    "xlam2-3b": (XLAM2MultiTurnMessages, False),
    "xlam2-1b": (XLAM2MultiTurnMessages, False),
    # other
    "gorilla": (GorillaMultiTurnMessages, False),
    "chatglm": (ChatGLMMultiTurnMessages, False),
    "fcm3.1": (FCMMultiTurnMessages, True),
    # Watt
    "watt70b": (WattMultiTurnMessages, True),
    "watt8b": (WattMultiTurnMessages, True),
    # Hammer
    "hammer7b": (HammerMultiTurnMessages, False),
    "hammer3b": (HammerMultiTurnMessages, False),
    "hammer1.5b": (HammerMultiTurnMessages, False),
    "hammer0.5b": (HammerMultiTurnMessages, False),
    # LLAMA
    "llama70b": (LlamaMultiTurnMessages, True),
    "llama8b": (LlamaMultiTurnMessages, True),
    "llama3b": (LlamaMultiTurnMessages, True),
    "llama1b": (LlamaMultiTurnMessages, True)
}

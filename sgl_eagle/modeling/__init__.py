from .auto_model import AutoModelForCausalLM
from .base.llama3 import LlamaForCausalLM
from .base.llama4 import Llama4ForCausalLM
from .draft.llama3_eagle import LlamaForCausalLMEagle3

__all__ = [
    "AutoModelForCausalLM",
    "LlamaForCausalLM",
    "Llama4ForCausalLM",
    "LlamaForCausalLMEagle3",
]

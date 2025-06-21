from transformers import LlamaForCausalLM, LlamaConfig


class LlamaForCausalLM(LlamaForCausalLM):

    def __init__(self, config: LlamaConfig):
        super().__init__(config)

    def forward(self, input_ids, attention_mask, loss_mask):
        pass


EntryClass = [LlamaForCausalLM]
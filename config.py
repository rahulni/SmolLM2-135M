# Auto-generated config for SmolLM2-135M
class SmolLMConfig:
    def __init__(self):
        self.model_type = 'llama'
        self.vocab_size = 49152
        self.hidden_size = 576
        self.intermediate_size = 1536
        self.num_hidden_layers = 30
        self.num_attention_heads = 9
        self.num_key_value_heads = 3
        self.max_position_embeddings = 8192
        self.rms_norm_eps = 1e-05
        self.rope_theta = 100000
        self.rope_scaling = None
        self.bos_token_id = 0
        self.eos_token_id = 0
        self.pad_token_id = None
        self.tie_word_embeddings = True

    def __repr__(self):
        return f"SmolLMConfig({vars(self)})"
from dataclasses import dataclass


@dataclass
class TranformerConfig:
    vocab_size: int
    context_length: int
    num_layers: int
    d_model: int
    num_heads: int
    # d_ff: int  # should be 4 * d_model


GPT_2_small_config = TranformerConfig(vocab_size=50257, context_length=1024, num_layers=12, d_model=768, num_heads=12)
GPT_2_medium_config = TranformerConfig(vocab_size=50257, context_length=1024, num_layers=24, d_model=1024, num_heads=16)
GPT_2_large_config = TranformerConfig(vocab_size=50257, context_length=1024, num_layers=36, d_model=1280, num_heads=20)
GPT_2_XL_config = TranformerConfig(vocab_size=50257, context_length=1024, num_layers=48, d_model=1600, num_heads=25)
GPT_2_XL_config_extra_context = TranformerConfig(
    vocab_size=50257, context_length=16384, num_layers=48, d_model=1600, num_heads=25
)

TinyStoreis_config = TranformerConfig(vocab_size=10_000, context_length=256, num_layers=4, d_model=512, num_heads=16)


class TransformerAccountant:
    def __init__(self, config, precision_in_bytes=4):
        self.config = config
        # Number of bytes to represent a parameter. We assume a single precision float32 = 4 bytes.
        self.precision_in_bytes = precision_in_bytes

    def get_number_trainable_parameters(self):
        cfg = self.config
        # (1) Embedding Layer
        embedding_parameters = cfg.vocab_size * cfg.d_model

        # (2) Blocks
        rms_1_parameters = cfg.d_model
        mh_attn_parameters = 4 * (cfg.d_model**2)
        rms_2_parameters = cfg.d_model
        d_ff = 4 * cfg.d_model
        ff_parameters = 3 * (cfg.d_model * d_ff)

        block_parameters = rms_1_parameters + mh_attn_parameters + rms_2_parameters + ff_parameters

        blocks_total_parameters = cfg.num_layers * block_parameters

        # (3) Outer layer norm
        outer_ln_parameters = cfg.d_model

        # (4) Final projection
        final_proj_parameters = cfg.vocab_size * cfg.d_model

        return embedding_parameters + blocks_total_parameters + outer_ln_parameters + final_proj_parameters

    def get_trainable_parameters_footprint(self):
        num_trainable_parameters = self.get_number_trainable_parameters()
        return self.precision_in_bytes * num_trainable_parameters

    def get_total_flops_forward_pass(self):
        cfg = self.config

        # (1) Blocks
        # (1.1) Attention
        attn_projs = 3 * (2 * cfg.context_length * cfg.d_model * cfg.d_model)
        attn_matrix = 2 * cfg.context_length * cfg.context_length * cfg.d_model
        attn_values = 2 * cfg.context_length * cfg.context_length * cfg.d_model
        attn_output = 2 * cfg.context_length * cfg.d_model * cfg.d_model
        total_attn = attn_projs + attn_matrix + attn_values + attn_output

        # (1.2) FF
        d_ff = 4 * cfg.d_model
        ff_ws = 3 * (2 * cfg.context_length * cfg.d_model * d_ff)

        print("---------------------------")
        print("Breakdown within Block.")
        print(f"Attention: {total_attn:,}")
        print(f"FFW: {ff_ws:,}")
        ratio = total_attn / (total_attn + ff_ws)
        print(f"Attention percentage of block: {ratio:.2}")
        print("---------------------------")

        total_blocks = cfg.num_layers * (total_attn + ff_ws)

        # (2) Final projection
        final_proj = 2 * cfg.context_length * cfg.d_model * cfg.vocab_size

        return total_blocks + final_proj

    def total_memory_footprint_formula(self):
        cfg = self.config

        # (1) Parameters

        # (1.1) Transformer Block
        rms_norms = 2 * cfg.d_model
        mh_attn = 4 * cfg.d_model * cfg.d_model
        ffw = 3 * (4 * cfg.d_model * cfg.d_model)
        block_total = rms_norms + mh_attn + ffw
        all_blocks_total = block_total * cfg.num_layers

        final_rms = cfg.d_model
        output_embedding = cfg.vocab_size * cfg.d_model

        total_parameters = all_blocks_total + final_rms + output_embedding

        # (2) Gradients
        total_gradients = total_parameters

        # (3) Optimizer State
        total_optimizer_state = 2 * total_parameters

        total_independent_of_batch_size = (
            total_parameters + total_gradients + total_optimizer_state
        ) * self.precision_in_bytes

        # (4) Activations (all depend on batch size)

        # (4.1) Transformer Block
        rms_norms = 2 * cfg.d_model * cfg.d_model
        mh_attn = 5 * cfg.context_length * cfg.d_model + cfg.context_length * cfg.context_length * cfg.num_heads
        ffw = 9 * cfg.context_length * cfg.d_model
        block_total = rms_norms + mh_attn + ffw
        all_blocks_total = block_total * cfg.num_layers

        final_rms = cfg.context_length * cfg.d_model
        output_embedding = cfg.context_length * cfg.vocab_size
        cross_entropy = cfg.context_length

        total_activations = all_blocks_total + final_rms + output_embedding + cross_entropy

        total_dependent_on_batch_size = total_activations * self.precision_in_bytes

        print(f"Number of parameters: {total_parameters:,}")

        print(f"Memory Footprint: {total_dependent_on_batch_size:,}B + {total_independent_of_batch_size:,}")


if __name__ == "__main__":
    gpt_2_xl = TransformerAccountant(GPT_2_XL_config)

    print(
        f"Trainable Parameters: {gpt_2_xl.get_number_trainable_parameters():,}",
    )
    print(f"Trainable Parameters Memory: {gpt_2_xl.get_trainable_parameters_footprint():,}")
    print(f"Matrix FLOPs required: {gpt_2_xl.get_total_flops_forward_pass():,}")

    all_configs = [
        ("gpt-small", GPT_2_small_config),
        ("gpt-medium", GPT_2_medium_config),
        ("gpt-large", GPT_2_large_config),
        ("gpt-xl", GPT_2_XL_config),
        ("gpt-xl-extra-context", GPT_2_XL_config_extra_context),
        ("tiny_stories", TinyStoreis_config),
    ]

    print()
    print()
    print()

    for name, config in all_configs:
        acct = TransformerAccountant(config)
        print(f"----------------- {name} --------------------")
        print(f"Matrix FLOPs required: {acct.get_total_flops_forward_pass():,}")
        acct.total_memory_footprint_formula()
        print("---------------------------------------------")
        print()

from cs336_basics.param_defs import LLM_Params, Optimizer_Params


# ---------------------   Debug Params -------------------------
debug_llm = LLM_Params(
    vocab_size=-1,  # set dynamically in the code since it varies based on the dataset
    context_length=64,
    num_layers=4,
    d_model=32,
    num_heads=4,
    d_ff=4 * 32,
    rope_theta=10_000,
)
debug_opt = Optimizer_Params(
    min_lr=1e-2,
    max_lr=1e-1,
    warmup_iters=1000,
    total_iters=10000,
    betas=(0.9, 0.95),
    weight_decay=0.9,
    eps=1e-8,
    max_norm=1e-2,
)

# ---------------------   Tiny Stories Initial -------------------------
tiny_llm = LLM_Params(
    vocab_size=10_000,  # provided
    context_length=256,  # provided
    num_layers=4,  # provided
    d_model=512,  # provided
    num_heads=16,  # provided
    d_ff=1344,  # provided (roughly 8/3 * d_mdodel but still multiple of 64)
    rope_theta=10_000,  # provided
)

# Initial params suggested by chatgpt.
tiny_opt = Optimizer_Params(
    min_lr=0.0,
    max_lr=5e-4,
    warmup_iters=500,
    total_iters=50_000,
    betas=(0.9, 0.95),
    weight_decay=0.05,
    eps=1e-8,
    max_norm=1.0,
)


# Populate this map with different config settings.
EXPERIMENTAL_CONFIGS = {"debug": (debug_llm, debug_opt), "tiny_llm_initial": (tiny_llm, tiny_opt)}

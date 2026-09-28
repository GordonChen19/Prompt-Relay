"""Prompt Relay and temporal sliding attention for MiniMax H3."""


def enable_prompt_relay(pipe):
    """Add per-request relay/sliding options to an upstream H3 ModularPipeline."""
    from .pipeline import enable_prompt_relay as install

    return install(pipe)

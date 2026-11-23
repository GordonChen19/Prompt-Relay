"""Prompt Relay integration for the official MiniMax H3 Diffusers pipeline."""

from .config import PromptRelayConfig

__all__ = ["PromptRelayConfig", "enable_prompt_relay"]


def enable_prompt_relay(pipe):
    """Lazily import the Diffusers integration so config validation stays light."""

    from .pipeline import enable_prompt_relay as _enable_prompt_relay

    return _enable_prompt_relay(pipe)

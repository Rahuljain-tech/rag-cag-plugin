"""Rough token estimates (chars / 4) for local models without a tokenizer."""


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_generation_tokens(query: str, context: str, response: str) -> int:
    """Estimate tokens that would be consumed by an LLM generate call."""
    template_overhead = 40
    return (
        estimate_tokens(query)
        + estimate_tokens(context)
        + estimate_tokens(response)
        + template_overhead
    )

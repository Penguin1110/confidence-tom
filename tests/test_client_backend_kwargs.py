from confidence_tom.infra.client import LLMClient


def test_ollama_completion_kwargs_include_ollama_options() -> None:
    client = LLMClient(
        model="qwen/qwen3-14b:nitro",
        backend="ollama",
        local_model_name="qwen3:14b",
        temperature=0.0,
        max_tokens=123,
        top_p=0.9,
        top_k=20,
        seed=7,
        num_ctx=8192,
        num_predict=456,
        enable_thinking=False,
    )

    kwargs = client._completion_kwargs()
    assert kwargs["model"] == "qwen3:14b"
    assert kwargs["temperature"] == 0.0
    assert kwargs["max_tokens"] == 123
    assert kwargs["top_p"] == 0.9
    assert kwargs["extra_body"]["options"] == {
        "top_p": 0.9,
        "top_k": 20,
        "seed": 7,
        "num_ctx": 8192,
        "num_predict": 456,
    }
    assert kwargs["extra_body"]["enable_thinking"] is False
    assert kwargs["extra_body"]["think"] is False


def test_openrouter_completion_kwargs_preserve_original_shape() -> None:
    client = LLMClient(
        model="openai/gpt-5.4",
        backend="openrouter",
        temperature=0.0,
        max_tokens=321,
        top_p=0.8,
        seed=11,
    )

    kwargs = client._completion_kwargs()
    assert kwargs["model"] == "openai/gpt-5.4"
    assert kwargs["temperature"] == 0.0
    assert kwargs["max_tokens"] == 321
    assert kwargs["top_p"] == 0.8
    assert kwargs["seed"] == 11
    assert "extra_body" not in kwargs


"""Tests for the Muse Spark API client."""

from types import SimpleNamespace

import pytest

from src.interpretation.muse_spark_client import (
    META_API_BASE_URL,
    MUSE_SPARK_MODEL,
    MuseSparkClient,
    MuseSparkClientError,
)


class FakeResponses:
    def __init__(self, response=None):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeSDKClient:
    def __init__(self, response=None):
        self.responses = FakeResponses(response=response)


def test_generate_task_definition_returns_model_output():
    expected_output = '{"task_summary": "Move the cube"}'
    sdk_client = FakeSDKClient(
        response=SimpleNamespace(output_text=expected_output)
    )
    client = MuseSparkClient(sdk_client)

    result = client.generate_task_definition("Move the cube.")

    assert result == expected_output
    assert len(sdk_client.responses.calls) == 1

    request = sdk_client.responses.calls[0]

    assert request["model"] == MUSE_SPARK_MODEL
    assert request["input"] == "Move the cube."
    assert request["instructions"]
    assert request["max_output_tokens"] == 2500
    assert request["store"] is False


def test_generate_task_definition_strips_instruction_and_output():
    sdk_client = FakeSDKClient(
        response=SimpleNamespace(output_text="  { }  ")
    )
    client = MuseSparkClient(sdk_client)

    result = client.generate_task_definition("  Move the cube.  ")

    assert result == "{ }"
    assert sdk_client.responses.calls[0]["input"] == "Move the cube."


@pytest.mark.parametrize("instruction", ["", " ", None, 123])
def test_generate_task_definition_rejects_invalid_instruction(
    instruction,
):
    client = MuseSparkClient(FakeSDKClient())

    with pytest.raises(ValueError, match="task_instruction"):
        client.generate_task_definition(instruction)


@pytest.mark.parametrize("output", [None, "", " ", 123])
def test_generate_task_definition_rejects_empty_or_invalid_output(output):
    sdk_client = FakeSDKClient(
        response=SimpleNamespace(output_text=output)
    )
    client = MuseSparkClient(sdk_client)

    with pytest.raises(MuseSparkClientError, match="empty or invalid"):
        client.generate_task_definition("Move the cube.")


def test_constructor_rejects_client_without_responses():
    with pytest.raises(TypeError, match="responses interface"):
        MuseSparkClient(object())


@pytest.mark.parametrize("model", ["", " "])
def test_constructor_rejects_empty_model(model):
    with pytest.raises(ValueError, match="model"):
        MuseSparkClient(FakeSDKClient(), model=model)


@pytest.mark.parametrize("max_tokens", [0, -1, True, 1.5])
def test_constructor_rejects_invalid_max_output_tokens(max_tokens):
    with pytest.raises(ValueError, match="max_output_tokens"):
        MuseSparkClient(
            FakeSDKClient(),
            max_output_tokens=max_tokens,
        )


def test_from_environment_uses_model_api_key(monkeypatch):
    captured = {}

    def fake_openai(**kwargs):
        captured.update(kwargs)
        return FakeSDKClient()

    monkeypatch.setenv("MODEL_API_KEY", "test-model-key")
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.setattr(
        "src.interpretation.muse_spark_client.OpenAI",
        fake_openai,
    )

    client = MuseSparkClient.from_environment()

    assert isinstance(client, MuseSparkClient)
    assert captured["base_url"] == META_API_BASE_URL
    assert captured["api_key"] == "test-model-key"
    assert captured["timeout"] == 60.0


def test_from_environment_falls_back_to_meta_api_key(monkeypatch):
    captured = {}

    def fake_openai(**kwargs):
        captured.update(kwargs)
        return FakeSDKClient()

    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.setenv("META_API_KEY", "fallback-key")
    monkeypatch.setattr(
        "src.interpretation.muse_spark_client.OpenAI",
        fake_openai,
    )

    MuseSparkClient.from_environment()

    assert captured["api_key"] == "fallback-key"


def test_from_environment_rejects_missing_api_key(monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("META_API_KEY", raising=False)

    with pytest.raises(MuseSparkClientError, match="must be configured"):
        MuseSparkClient.from_environment()


def test_from_environment_rejects_non_positive_timeout(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "test-key")

    with pytest.raises(ValueError, match="timeout_seconds"):
        MuseSparkClient.from_environment(timeout_seconds=0)

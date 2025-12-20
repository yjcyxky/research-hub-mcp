import pytest

from text2table.text2table import AsyncText2Table


class _DummyAsyncVLLM:
    def __init__(self, output: str):
        self.output = output
        self.calls = 0

    async def generate(self, *args, **kwargs) -> str:  # pragma: no cover - trivial passthrough
        self.calls += 1
        return self.output

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_async_run_without_gliner(monkeypatch):
    client = _DummyAsyncVLLM("| A |\n| --- |\n| x |")
    extractor = AsyncText2Table(
        labels=["A"],
        use_gliner=False,
        server_url="http://localhost:8000/v1",
    )

    async def fake_client():
        return client

    monkeypatch.setattr(extractor, "_get_vllm_client", fake_client)

    table, entities = await extractor.run("some text")

    assert "| A |" in table
    assert entities == []
    assert client.calls == 1


@pytest.mark.asyncio
async def test_async_run_with_thinking(monkeypatch):
    output = "<think>step1</think>\n| Drug |\n| --- |\n| R13 |"
    client = _DummyAsyncVLLM(output)
    extractor = AsyncText2Table(
        labels=["Drug"],
        use_gliner=False,
        enable_thinking=True,
        server_url="http://localhost:8000/v1",
    )

    async def fake_client():
        return client

    monkeypatch.setattr(extractor, "_get_vllm_client", fake_client)

    table, entities = await extractor.run("example text")
    thinking, final_table, _ = await extractor.run_with_thinking("example text")

    assert table.startswith("| Drug |")
    assert final_table.startswith("| Drug |")
    assert "step1" in thinking
    assert entities == []
    assert client.calls == 2

import pytest
from production_rag.context import ContextOptimizer
from production_rag.tools import SafeCalculator, ToolRouter


def make_chunks():
    return [{"chunk_id": f"c{i}", "document_id": f"d{i}", "source": f"s{i}",
             "text": ("The cloud infrastructure budget allocates 45 percent to generative AI. " * (i + 2)),
             "score": 1 / (i + 1)} for i in range(8)]


def test_context_optimizer_respects_total_token_budget():
    optimizer = ContextOptimizer(token_budget=1200, max_candidates=20, max_passages=5,
                                 generation_reserve=100, evaluation_reserve=50)
    package = optimizer.optimize("cloud infrastructure generative AI budget", make_chunks(), "system prompt", "user query")
    assert package.token_count <= 1200
    assert len(package.chunks) <= 5


def test_context_optimizer_keeps_source_identifiers():
    package = ContextOptimizer(token_budget=2500).optimize("cloud infrastructure budget", make_chunks(), "system")
    assert "source_id=" in package.text


def test_context_optimizer_handles_empty_retrieval():
    package = ContextOptimizer(token_budget=500, generation_reserve=100, evaluation_reserve=100).optimize("question", [], "system")
    assert package.text == ""
    assert package.token_count <= 500


def test_calculator_handles_basic_arithmetic():
    assert SafeCalculator().calculate("(125 * 4) / 5") == 100.0


@pytest.mark.parametrize("expression", ["__import__('os').system('whoami')", "open('/etc/passwd')", "2 ** 99"])
def test_calculator_rejects_unsafe_expressions(expression):
    with pytest.raises((ValueError, TypeError)):
        SafeCalculator().calculate(expression)


def test_tool_router_does_not_call_external_search_by_default():
    import asyncio
    results, tools = asyncio.run(ToolRouter().route("What is the policy?", allow_external=False))
    assert results == []
    assert tools == []


def test_tool_router_uses_calculator_for_expression():
    import asyncio
    results, tools = asyncio.run(ToolRouter().route("calculate: (125 * 4) / 5"))
    assert tools == ["calculator"]
    assert "100.0" in results[0]["content"]


def test_cross_encoder_falls_back_within_configured_budget():
    import time
    class SlowCrossEncoder:
        def predict(self, pairs):
            time.sleep(0.25)
            return [0.9] * len(pairs)
    optimizer = ContextOptimizer(token_budget=2000, generation_reserve=100, evaluation_reserve=100,
                                 reranker_backend="cross-encoder", cross_encoder=SlowCrossEncoder(),
                                 rerank_timeout_seconds=0.03)
    package = optimizer.optimize("cloud infrastructure", make_chunks(), "system")
    assert package.reranker_backend == "lexical_timeout_fallback"
    assert package.rerank_ms < 150


def test_external_search_failure_does_not_stop_pipeline_tools():
    import asyncio
    class FailingSearch:
        async def search(self, query, max_results=5):
            raise TimeoutError("provider unavailable")
    results, tools = asyncio.run(ToolRouter(external_search=FailingSearch()).route(
        "Find the latest company policy", allow_external=True))
    assert results == []
    assert tools == ["tavily_unavailable"]

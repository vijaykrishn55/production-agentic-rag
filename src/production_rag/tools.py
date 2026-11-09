from __future__ import annotations

import ast
import operator
import hashlib
from typing import Any


_ALLOWED_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


class SafeCalculator:
    def calculate(self, expression: str) -> float:
        if len(expression) > 160:
            raise ValueError("Expression too long")
        tree = ast.parse(expression, mode="eval")
        def visit(node: ast.AST) -> float:
            if isinstance(node, ast.Expression):
                return visit(node.body)
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                return float(node.value)
            if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
                left, right = visit(node.left), visit(node.right)
                if isinstance(node.op, ast.Pow) and abs(right) > 8:
                    raise ValueError("Exponent outside allowed range")
                value = _ALLOWED_OPERATORS[type(node.op)](left, right)
                if abs(value) > 1e15:
                    raise ValueError("Result outside allowed range")
                return float(value)
            if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
                return float(_ALLOWED_OPERATORS[type(node.op)](visit(node.operand)))
            raise ValueError("Only basic arithmetic expressions are allowed")
        return visit(tree)


class TavilySearch:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def search(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
        if not self.api_key:
            raise RuntimeError("TAVILY_API_KEY is not configured")
        import httpx
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={"api_key": self.api_key, "query": query, "max_results": max_results,
                      "search_depth": "basic", "include_answer": False},
            )
            response.raise_for_status()
            payload = response.json()
        return [{"title": item.get("title", ""), "url": item.get("url", ""),
                 "content": item.get("content", "")} for item in payload.get("results", [])]


class ToolRouter:
    def __init__(self, calculator: SafeCalculator | None = None, external_search: TavilySearch | None = None) -> None:
        self.calculator = calculator or SafeCalculator()
        self.external_search = external_search

    async def route(self, query: str, allow_external: bool = False) -> tuple[list[dict[str, Any]], list[str]]:
        tools_used: list[str] = []
        results: list[dict[str, Any]] = []
        stripped = query.strip()
        candidate = stripped.removeprefix("calculate:").strip()
        if candidate and len(candidate) < 100 and all(ch in "0123456789.+-*/()% \t" for ch in candidate):
            try:
                value = self.calculator.calculate(candidate)
                results.append({"tool": "calculator", "source_id": "tool:calculator", "source": "calculator",
                                "document_id": "calculator", "content": f"Calculation result: {value}"})
                tools_used.append("calculator")
            except (ValueError, ZeroDivisionError, SyntaxError):
                pass
        if allow_external and self.external_search:
            try:
                for item in await self.external_search.search(query):
                    source_id = "external:" + hashlib.sha256(item.get("url", "").encode()).hexdigest()[:12]
                    results.append({"tool": "tavily", "source_id": source_id,
                                    "source": item.get("url", "Tavily search"),
                                    "document_id": "external-search", **item})
                tools_used.append("tavily")
            except Exception:
                # External search is an optional tool; continue with internal evidence.
                tools_used.append("tavily_unavailable")
        return results, tools_used

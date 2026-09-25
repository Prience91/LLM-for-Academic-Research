from __future__ import annotations

import unittest

from giis_prod.indexing import GraniteEmbeddingFunction


class _Encoded:
    def __init__(self, rows: list[list[float]]):
        self.rows = rows

    def tolist(self) -> list[list[float]]:
        return self.rows


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], bool]] = []

    def encode(self, inputs: list[str], *, normalize_embeddings: bool) -> _Encoded:
        self.calls.append((inputs, normalize_embeddings))
        return _Encoded([[0.25, 0.75] for _ in inputs])


class GraniteEmbeddingCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.embedding = GraniteEmbeddingFunction.__new__(GraniteEmbeddingFunction)
        self.embedding.model_name = "fake"
        self.embedding.model = _FakeModel()

    def test_callable_embedding_is_normalized(self) -> None:
        result = self.embedding(["document"])

        self.assertEqual(result, [[0.25, 0.75]])
        self.assertEqual(self.embedding.model.calls, [(["document"], True)])

    def test_embed_query_delegates_to_same_normalized_path(self) -> None:
        result = self.embedding.embed_query(["query"])

        self.assertEqual(result, [[0.25, 0.75]])
        self.assertEqual(self.embedding.model.calls, [(["query"], True)])


if __name__ == "__main__":
    unittest.main()

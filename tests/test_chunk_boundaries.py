from __future__ import annotations

import unittest

from giis_prod.ingestion import _bounded_token_windows


class _WhitespaceTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[str]:
        del add_special_tokens
        return text.split()

    def decode(self, tokens: list[str], *, skip_special_tokens: bool) -> str:
        del skip_special_tokens
        return " ".join(tokens)


class ChunkBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = _WhitespaceTokenizer()

    def test_oversized_text_is_split_without_losing_tokens(self):
        tokens = [f"token{index}" for index in range(25)]
        windows = _bounded_token_windows(" ".join(tokens), self.tokenizer, 10, 2)
        self.assertGreater(len(windows), 1)
        self.assertTrue(all(count <= 10 for _, count in windows))
        recovered = {token for text, _ in windows for token in text.split()}
        self.assertEqual(recovered, set(tokens))

    def test_text_at_boundary_remains_a_single_window(self):
        text = " ".join(f"token{index}" for index in range(10))
        self.assertEqual(_bounded_token_windows(text, self.tokenizer, 10, 2), [(text, 10)])

    def test_overlap_must_be_smaller_than_window(self):
        with self.assertRaises(ValueError):
            _bounded_token_windows("one two", self.tokenizer, 2, 2)


if __name__ == "__main__":
    unittest.main()

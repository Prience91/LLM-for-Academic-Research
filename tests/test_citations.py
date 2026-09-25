from __future__ import annotations

import unittest

from giis_prod.citations import (
    citation_coverage,
    contains_non_corpus_citation,
    doi_url,
    inline_citation,
    normalized_doi,
    render_citation_popovers,
    render_inline_citations,
    render_reference_panel,
    validate_markers,
)
from giis_prod.models import Evidence


class CitationTests(unittest.TestCase):
    def setUp(self):
        self.first = Evidence(
            evidence_id="E1",
            chunk_id="c1",
            document_id="d1",
            text="Text",
            title="Paper",
            authors=["Ada Researcher", "Ben Scientist"],
            year=2024,
            journal="Journal",
            page_start=2,
            page_end=3,
            doi="https://doi.org/10.1234/Example.1",
            dense_similarity=0.81,
        )
        self.second = Evidence(
            evidence_id="E2",
            chunk_id="c2",
            document_id="d1",
            text="More text",
            title="Paper",
            authors=["Ada Researcher", "Ben Scientist"],
            year=2024,
            journal="Journal",
            page_start=8,
            page_end=8,
            doi="10.1234/Example.1",
            dense_similarity=0.68,
        )

    def test_markers_are_allowlisted(self):
        self.assertTrue(validate_markers("Supported claim [E1].", [self.first])[0])
        self.assertFalse(validate_markers("Invented marker [E9].", [self.first])[0])

    def test_rendering_is_deterministic(self):
        self.assertEqual(inline_citation(self.first), "(Researcher & Scientist, 2024)")
        self.assertEqual(
            render_inline_citations("Supported [E1].", [self.first]),
            "Supported (Researcher & Scientist, 2024).",
        )
        self.assertNotIn("E1", render_inline_citations("Supported [E1].", [self.first]))

    def test_adjacent_sources_share_one_plain_apa_parenthesis(self):
        other = Evidence(
            evidence_id="E3",
            chunk_id="c3",
            document_id="d2",
            text="Other text",
            title="Other Paper",
            authors=["Cara Analyst"],
            year=2021,
            journal="Other Journal",
            page_start=4,
            page_end=4,
        )
        rendered = render_inline_citations(
            "Combined support [E1] [E2]; [E3].", [self.first, self.second, other]
        )

        self.assertTrue(rendered.startswith("Combined support ("))
        self.assertTrue(rendered.endswith(")."))
        self.assertIn("Researcher & Scientist, 2024", rendered)
        self.assertIn("Analyst, 2021", rendered)
        self.assertIn("; ", rendered)
        self.assertNotIn("](#giis-source-", rendered)

    def test_doi_is_normalized_only_when_available_and_valid(self):
        self.assertEqual(
            normalized_doi("https://doi.org/10.1234/Example.1"),
            "10.1234/Example.1",
        )
        self.assertEqual(
            doi_url("doi:10.1234/a value"),
            None,
        )
        self.assertIsNone(normalized_doi(None))

    def test_reference_panel_consolidates_document_pages(self):
        panel = render_reference_panel([self.first, self.second])
        self.assertEqual(panel.count("### Paper"), 1)
        self.assertIn("2-3, 8", panel)
        self.assertIn("[10.1234/Example.1](https://doi.org/10.1234/Example.1)", panel)
        self.assertIn("Similarity score:** `81.0%`", panel)
        self.assertIn("🟢 High relevance", panel)
        self.assertNotIn("Document ID", panel)
        self.assertNotIn("Retrieval score", panel)

    def test_similarity_badges_match_original_cosine_thresholds(self):
        cases = (
            (0.75, "🟢 High relevance"),
            (0.65, "🟡 Moderate relevance"),
            (0.64, "🟠 Marginal relevance"),
        )
        for similarity, expected in cases:
            with self.subTest(similarity=similarity):
                evidence = Evidence(
                    evidence_id="E1",
                    chunk_id=f"c-{similarity}",
                    document_id=f"d-{similarity}",
                    text="Text",
                    title="Paper",
                    authors=[],
                    year=2024,
                    journal=None,
                    page_start=1,
                    page_end=1,
                    dense_similarity=similarity,
                )
                panel = render_reference_panel([evidence])
                self.assertIn(f"`{similarity * 100:.1f}%`", panel)
                self.assertIn(expected, panel)

    def test_similarity_is_not_invented_when_dense_score_is_absent(self):
        evidence = Evidence(
            evidence_id="E1",
            chunk_id="c-none",
            document_id="d-none",
            text="Text",
            title="Paper",
            authors=[],
            year=2024,
            journal=None,
            page_start=1,
            page_end=1,
        )
        self.assertIn(
            "Similarity score:** Not available for this retrieval",
            render_reference_panel([evidence]),
        )

    def test_coverage_counts_factual_sentences(self):
        text = (
            "This sufficiently long factual sentence is supported by corpus evidence [E1]. "
            "This second sufficiently long factual sentence lacks a citation marker entirely."
        )
        self.assertEqual(citation_coverage(text), 0.5)

    def test_coverage_counts_short_cited_facts(self):
        self.assertEqual(citation_coverage("Supported [E1]."), 1.0)
        self.assertEqual(citation_coverage("4.2% [E1]."), 1.0)

    def test_coverage_counts_markdown_table_rows_and_ignores_structure(self):
        text = (
            "| Attribute | Arabica | Robusta |\n"
            "|---|---|---|\n"
            "| Heat sensitivity | Higher [E1] | Lower [E1] |\n"
            "| Climate variability | Moderate [E2] | Restrictive [E2] |"
        )
        self.assertEqual(citation_coverage(text), 1.0)

    def test_coverage_rejects_uncited_markdown_table_claims(self):
        text = (
            "| Attribute | Arabica | Robusta |\n"
            "|:---|:---:|---:|\n"
            "| Heat sensitivity | Higher [E1] | Lower [E1] |\n"
            "| Climate variability | Moderate | Restrictive |"
        )
        self.assertEqual(citation_coverage(text), 0.5)

    def test_coverage_does_not_count_marker_only_or_short_uncited_fragments(self):
        self.assertEqual(citation_coverage("[E1]."), 0.0)
        self.assertEqual(citation_coverage("Supported."), 0.0)

    def test_non_corpus_citation_detection_rejects_nested_bibliography_styles(self):
        for text in (
            "Smith et al. (2020) reported the outcome [E1].",
            "The outcome was previously reported (Smith, 2020) [E1].",
            "The review referred to earlier work [12] [E1].",
            "Several sources were listed [4, 7] [E1].",
        ):
            with self.subTest(text=text):
                self.assertTrue(contains_non_corpus_citation(text))
        self.assertFalse(
            contains_non_corpus_citation(
                "The ingested corpus paper reported the outcome [E1]."
            )
        )

    def test_popovers_are_real_metadata_backed_and_consolidated_by_document(self):
        rendered = render_citation_popovers([self.first, self.second])
        self.assertNotIn('class="giis-citation-chip"', rendered)
        self.assertNotIn("Evidence IDs", rendered)
        self.assertIn("Ada Researcher, Ben Scientist", rendered)
        self.assertIn("Journal", rendered)
        self.assertIn("2-3, 8", rendered)
        self.assertIn('class="giis-citation-modal"', rendered)
        self.assertIn("background: #050505", rendered)
        self.assertIn("color: #fff", rendered)
        self.assertIn("<dt>DOI</dt>", rendered)
        self.assertIn('href="https://doi.org/10.1234/Example.1"', rendered)
        self.assertNotIn("Corpus document ID", rendered)
        self.assertNotIn("Retrieval score", rendered)

    def test_popover_escapes_metadata_and_never_invents_missing_authors(self):
        evidence = Evidence(
            evidence_id="E1",
            chunk_id="chunk",
            document_id="doc<script>",
            text="Evidence",
            title="Paper <script>alert(1)</script>",
            authors=[],
            year=None,
            journal=None,
            page_start=1,
            page_end=1,
        )
        rendered = render_citation_popovers([evidence])
        self.assertNotIn("<script>alert", rendered)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", rendered)
        self.assertIn("title used; author metadata unavailable", rendered)
        self.assertIn("Not available in corpus metadata", rendered)
        self.assertNotIn("<dt>DOI</dt>", rendered)

    def test_reader_facing_rendering_normalizes_stroked_z_glyphs(self):
        evidence = Evidence(
            evidence_id="E1",
            chunk_id="chunk",
            document_id="doc-z",
            text="Evidence",
            title="Ƶambia and ƶebra",
            authors=[],
            year=2024,
            journal=None,
            page_start=1,
            page_end=1,
        )

        rendered_answer = render_inline_citations("Ƶone [E1].", [evidence])
        rendered_panel = render_reference_panel([evidence])
        rendered_modal = render_citation_popovers([evidence])

        for rendered in (rendered_answer, rendered_panel, rendered_modal):
            self.assertNotIn("Ƶ", rendered)
            self.assertNotIn("ƶ", rendered)


if __name__ == "__main__":
    unittest.main()

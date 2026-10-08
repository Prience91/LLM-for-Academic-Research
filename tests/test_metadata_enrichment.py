from __future__ import annotations

import unittest
import urllib.error
from email.message import Message
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from giis_prod.config import Settings
from giis_prod.ingestion import _native_text_is_adequate, _raw_pdf_year, _select_title
from giis_prod.metadata import (
    MetadataResolver,
    MetadataQuotaDeferred,
    normalize_doi,
    title_relation,
    title_similarity,
)


def _settings(root: Path) -> Settings:
    return Settings(
        root=root,
        data={
            "paths": {"metadata_cache": "artifacts/metadata/cache"},
            "metadata": {
                "enabled": True,
                "allow_network": False,
                "crossref_enabled": True,
                "openalex_enabled": True,
                "request_timeout_seconds": 1,
                "maximum_retries": 0,
                "request_delay_seconds": 0.0,
                "minimum_title_similarity": 0.93,
                "minimum_ambiguity_margin": 0.03,
                "minimum_title_fragment_tokens": 6,
                "minimum_title_fragment_characters": 40,
                "user_agent": "test",
                "contact_email": "",
                "filename_is_curated_title": True,
            },
        },
    )


class MetadataEnrichmentTests(unittest.TestCase):
    def test_curated_filename_supersedes_generic_embedded_title(self):
        title, warnings = _select_title(
            "A Global Safety Net to reverse biodiversity loss",
            "Science Journals â€” AAAS",
            True,
        )
        self.assertEqual(title, "A Global Safety Net to reverse biodiversity loss")
        self.assertIn("embedded_title_disagrees_with_curated_filename", warnings)

    def test_corpus_identity_prefix_is_removed_before_metadata_lookup(self):
        title, warnings = _select_title(
            "034839e995f3__Variability and climate change trend in vegetation phenology",
            None,
            True,
        )
        self.assertEqual(
            title,
            "Variability and climate change trend in vegetation phenology",
        )
        self.assertIn("corpus_filename_hash_prefix_removed", warnings)

    def test_non_identity_prefix_is_preserved(self):
        title, warnings = _select_title(
            "study2026xyz__Climate adaptation evidence",
            None,
            True,
        )
        self.assertEqual(title, "study2026xyz__Climate adaptation evidence")
        self.assertNotIn("corpus_filename_hash_prefix_removed", warnings)

    def test_malformed_pdf_dates_are_read_without_pypdf_conversion(self):
        self.assertEqual(_raw_pdf_year({"/CreationDate": "27th February 2013"}), 2013)
        self.assertEqual(_raw_pdf_year({"/CreationDate": "12/15/2003 11:33:36"}), 2003)
        self.assertIsNone(_raw_pdf_year({"/CreationDate": "unknown"}))

    def test_doi_normalization(self):
        self.assertEqual(
            normalize_doi("https://doi.org/10.1234/ABC.7."), "10.1234/abc.7"
        )

    def test_title_similarity_is_normalization_aware(self):
        self.assertGreater(
            title_similarity("Climate-change: Risks", "Climate Change Risks"), 0.99
        )

    def test_distinctive_title_fragment_is_verified_truncation(self):
        self.assertEqual(
            title_relation(
                "and scientific capacity at a country level Disaggregation",
                "The relationship among biodiversity, governance, wealth, and scientific "
                "capacity at a country level: Disaggregation and prioritization",
            ),
            "verified_truncation",
        )

    def test_unrelated_titles_remain_a_conflict(self):
        self.assertEqual(
            title_relation(
                "Effects of meteorological factors on hemorrhagic fever",
                "A 38-year-old woman with zosteriform skin lesions",
            ),
            "conflict",
        )

    def test_native_text_adequacy_requires_both_depth_and_multiple_chunks(self):
        self.assertTrue(_native_text_is_adequate(["a" * 600, "b" * 500], 1000, 2))
        self.assertFalse(_native_text_is_adequate(["a" * 1100], 1000, 2))
        self.assertFalse(_native_text_is_adequate(["a" * 400, "b" * 400], 1000, 2))

    def test_verified_doi_record_overrides_weak_pdf_metadata(self):
        with TemporaryDirectory() as temporary:
            resolver = MetadataResolver(_settings(Path(temporary)))
            record = {
                "title": "Climate Risk and Coffee",
                "authors": ["A. Author"],
                "year": 2024,
                "journal": "Climate Journal",
                "publisher": "Verified Publisher",
                "volume": "4",
                "issue": "2",
                "doi": "10.1234/test",
                "document_type": "journal-article",
            }
            with patch.object(resolver, "_crossref_doi", return_value=record):
                result = resolver.enrich(
                    {
                        "title": "Climate Risk and Coffee",
                        "authors": [],
                        "year": None,
                        "doi": None,
                    },
                    "Climate Risk and Coffee DOI 10.1234/test",
                    {},
                )
        self.assertTrue(result.verified)
        self.assertEqual(result.values["authors"], ["A. Author"])
        self.assertEqual(result.provenance["authors"], "crossref_doi")

    def test_doi_match_cannot_override_a_conflicting_curated_title(self):
        with TemporaryDirectory() as temporary:
            resolver = MetadataResolver(_settings(Path(temporary)))
            wrong_record = {
                "title": "A 38-year-old woman with zosteriform skin lesions",
                "authors": ["Wrong Author"],
                "year": 2017,
                "doi": "10.1371/j",
                "document_type": "article",
            }
            with (
                patch.object(resolver, "_crossref_doi", return_value=wrong_record),
                patch.object(resolver, "_openalex_doi", return_value=None),
                patch.object(resolver, "_crossref_title", return_value=[]),
                patch.object(resolver, "_openalex_title", return_value=[]),
            ):
                result = resolver.enrich(
                    {
                        "title": "Effects of meteorological factors on hemorrhagic fever",
                        "authors": [],
                        "year": 2021,
                        "doi": None,
                    },
                    "DOI 10.1371/j",
                    {},
                )
        self.assertFalse(result.verified)
        self.assertEqual(result.match_source, "unresolved")
        self.assertTrue(
            any(warning.startswith("doi_title_mismatch:") for warning in result.warnings)
        )

    def test_doi_match_expands_a_distinctively_truncated_filename(self):
        full_title = (
            "The relationship among biodiversity, governance, wealth, and scientific "
            "capacity at a country level: Disaggregation and prioritization"
        )
        with TemporaryDirectory() as temporary:
            resolver = MetadataResolver(_settings(Path(temporary)))
            record = {
                "title": full_title,
                "authors": ["A. Author"],
                "year": 2020,
                "doi": "10.1234/full",
                "document_type": "journal-article",
            }
            with patch.object(resolver, "_crossref_doi", return_value=record):
                result = resolver.enrich(
                    {
                        "title": "and scientific capacity at a country level Disaggregation",
                        "authors": [],
                        "year": 2020,
                        "doi": None,
                    },
                    "DOI 10.1234/full",
                    {},
                )
        self.assertTrue(result.verified)
        self.assertEqual(result.values["title"], full_title)
        self.assertIn(
            "verified_filename_title_truncation:crossref_doi", result.warnings
        )

    def test_openalex_api_key_is_attached_to_requests(self):
        with TemporaryDirectory() as temporary:
            settings = _settings(Path(temporary))
            settings.data["metadata"]["openalex_api_key"] = "project-key"
            resolver = MetadataResolver(settings)
            captured = {}
            def fake(provider, key, url):
                captured["url"] = url
                return {"results": []}
            with patch.object(resolver, "_cached_json", side_effect=fake):
                resolver._openalex_title("Climate adaptation")
        self.assertIn("api_key=project-key", captured["url"])

    def test_rate_limit_opens_provider_circuit_breaker(self):
        with TemporaryDirectory() as temporary:
            settings = _settings(Path(temporary))
            settings.data["metadata"]["allow_network"] = True
            resolver = MetadataResolver(settings)
            headers = Message(); headers["Retry-After"] = "3600"
            error = urllib.error.HTTPError("https://example.test", 429, "limited", headers, None)
            with patch("urllib.request.urlopen", side_effect=error) as request:
                with self.assertRaises(MetadataQuotaDeferred):
                    resolver._cached_json("openalex", "one", "https://example.test/one")
                with self.assertRaises(MetadataQuotaDeferred):
                    resolver._cached_json("openalex", "two", "https://example.test/two")
        self.assertEqual(request.call_count, 1)

    def test_persisted_daily_budget_floor_blocks_only_billable_openalex_search(self):
        with TemporaryDirectory() as temporary:
            settings = _settings(Path(temporary))
            settings.data["metadata"]["allow_network"] = True
            resolver = MetadataResolver(settings)
            state = Path(temporary) / "artifacts/metadata/cache/openalex/rate_limit_state.json"
            state.parent.mkdir(parents=True)
            state.write_text(
                '{"reason":"daily_budget_floor","blocked_until_epoch":9999999999}',
                encoding="utf-8",
            )
            with patch("urllib.request.urlopen", side_effect=OSError("offline")) as request:
                with self.assertRaises(MetadataQuotaDeferred):
                    resolver._cached_json(
                        "openalex", "title:climate", "https://example.test/search"
                    )
                self.assertIsNone(
                    resolver._cached_json(
                        "openalex", "doi:10.1/test", "https://example.test/work"
                    )
                )
        self.assertEqual(request.call_count, 1)

    def test_rejected_project_key_is_recorded_without_secret(self):
        with TemporaryDirectory() as temporary:
            settings = _settings(Path(temporary))
            settings.data["metadata"]["allow_network"] = True
            settings.data["metadata"]["openalex_api_key"] = "secret-value"
            resolver = MetadataResolver(settings)
            error = urllib.error.HTTPError(
                "https://example.test", 401, "unauthorized", Message(), None
            )
            with patch("urllib.request.urlopen", side_effect=error):
                with self.assertRaises(MetadataQuotaDeferred):
                    resolver._cached_json(
                        "openalex", "title:climate", "https://example.test/search"
                    )
            state = (
                Path(temporary)
                / "artifacts/metadata/cache/openalex/rate_limit_state.json"
            ).read_text()
        self.assertIn("project_key_rejected", state)
        self.assertNotIn("secret-value", state)

    def test_unresolved_record_requires_review(self):
        with TemporaryDirectory() as temporary:
            resolver = MetadataResolver(_settings(Path(temporary)))
            result = resolver.enrich(
                {"title": "Unmatched report", "authors": [], "year": 2020}, "", {}
            )
        self.assertFalse(result.verified)
        self.assertEqual(result.match_source, "unresolved")
        self.assertIn("metadata_review_required", result.warnings)

    def test_verified_institutional_override_uses_corporate_author(self):
        with TemporaryDirectory() as temporary:
            resolver = MetadataResolver(_settings(Path(temporary)))
            result = resolver.enrich(
                {"title": "National adaptation report", "authors": [], "year": 2010},
                "",
                {
                    "verified": True,
                    "publisher": "Ministry of Environment",
                    "document_type": "report",
                },
            )
        self.assertTrue(result.verified)
        self.assertEqual(result.values["authors"], ["Ministry of Environment"])
        self.assertIsNone(result.values.get("doi"))
        self.assertEqual(result.provenance["authors"], "manual_verified_override")


if __name__ == "__main__":
    unittest.main()

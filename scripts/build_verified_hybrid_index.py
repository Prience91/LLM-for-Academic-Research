from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src'
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from giis_prod.config import Settings
from giis_prod.corpus_audit import CorpusAuditor
from giis_prod.indexing import build_indexes
from giis_prod.io import read_jsonl, sha256_file, write_json_atomic
from giis_prod.v17_turbo_contract import validate_retrieval_index_policy


def vector_payload_digest(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for path in sorted(
        value for value in root.rglob('*')
        if value.is_file() and value.name != 'chroma.sqlite3'
    ):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(b'\0')
        digest.update(sha256_file(path).encode())
        digest.update(b'\0')
        count += 1
    return digest.hexdigest(), count


def main() -> None:
    parser = argparse.ArgumentParser(description='Build the bound V17-Turbo hybrid index.')
    parser.add_argument('--config', default='config/settings_v17_turbo.toml')
    parser.add_argument('--expected-documents', type=int, required=True)
    parser.add_argument('--expected-chunks', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()

    settings = Settings.load(ROOT / args.config)
    status_path = settings.path('paths.artifact_dir') / 'logs/index_build_status.json'
    started_at = datetime.now(UTC).isoformat()
    write_json_atomic(status_path, {
        'status': 'running', 'started_at': started_at,
        'expected_documents': args.expected_documents,
        'expected_chunks': args.expected_chunks,
    })
    try:
        documents = list(read_jsonl(settings.path('paths.document_registry')))
        chunks = list(read_jsonl(settings.path('paths.chunk_file')))
        if len(documents) != args.expected_documents or len(chunks) != args.expected_chunks:
            raise ValueError(
                f'Frozen registry count mismatch: documents={len(documents)}, chunks={len(chunks)}'
            )
        pre_audit = CorpusAuditor(settings).run(check_indexes=False)
        manifest = build_indexes(settings, batch_size=args.batch_size)

        binding_path = settings.path('paths.index_binding')
        manifest.update({
            'binding_manifest': str(binding_path),
            'runtime_dependency_on_v17_index': False,
        })
        write_json_atomic(settings.path('paths.active_index'), manifest)
        post_audit = CorpusAuditor(settings).run(check_indexes=True)

        chroma_dir = Path(manifest['chroma_dir']).resolve()
        lexical_path = Path(manifest['lexical_db']).resolve()
        vector_digest, vector_file_count = vector_payload_digest(chroma_dir)
        binding = {
            'policy': 'v17-turbo',
            'schema_version': 1,
            'status': 'active',
            'created_at': datetime.now(UTC).isoformat(),
            'corpus_version': manifest['corpus_version'],
            'document_count': len(documents),
            'chunk_count': len(chunks),
            'embedding_model': manifest['embedding_model'],
            'collection_name': manifest['collection_name'],
            'document_registry_sha256': sha256_file(settings.path('paths.document_registry')),
            'chunk_registry_sha256': sha256_file(settings.path('paths.chunk_file')),
            'chroma_vector_payload_sha256': vector_digest,
            'chroma_vector_file_count': vector_file_count,
            'chroma_sqlite_integrity': 'runtime_collection_identity_and_count_audit',
            'lexical_db_sha256': sha256_file(lexical_path),
            'source_active_index_sha256': sha256_file(settings.path('paths.active_index')),
            'runtime_dependency_on_v17_index': False,
        }
        validate_retrieval_index_policy(
            settings.section('v17_turbo_retrieval_index'), binding
        )
        write_json_atomic(binding_path, binding)
        completed_at = datetime.now(UTC).isoformat()
        write_json_atomic(status_path, {
            'status': 'complete', 'started_at': started_at, 'completed_at': completed_at,
            'document_count': len(documents), 'chunk_count': len(chunks),
            'collection_name': manifest['collection_name'],
            'corpus_version': manifest['corpus_version'],
            'pre_index_audit_passed': pre_audit['passed'],
            'post_index_audit_passed': post_audit['passed'],
            'binding_manifest': str(binding_path),
        })
        print(json.dumps({'index': manifest, 'binding': binding, 'audit': post_audit}, indent=2))
    except Exception as exc:
        write_json_atomic(status_path, {
            'status': 'failed', 'started_at': started_at,
            'failed_at': datetime.now(UTC).isoformat(),
            'error': f'{type(exc).__name__}: {exc}',
        })
        raise


if __name__ == '__main__':
    main()

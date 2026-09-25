# Configuration

Copy `settings.example.toml` to `settings_v17_turbo.toml` for local execution. The latter is ignored from published deployments when it contains machine-specific changes.

Configuration values can be overridden through environment variables using the `GIIS__SECTION__KEY` form. Credentials must remain in environment variables and must not be written into TOML, manifests, logs, or artifacts.

The included values reproduce the V17-Turbo architecture and its DGX Spark serving profile. Corpus sizes and release paths are examples; a new run must create and bind its own corpus selection, split, index, semantic-review, and dataset-release manifests.


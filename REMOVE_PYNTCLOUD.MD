# Goal

* Remove the `pyntcloud` dependency (unmaintained and blocks pandas 3 migration).
* Keep or improve file IO behavior for `las`, `laz`, `csv`, `xyz`, and `pcd`.
* Keep tests stable: once added, migration work should make them pass without rewriting test expectations.

## File import/export strategy

* `las`/`laz`: `laspy`
* `csv`: `pandas`
* `xyz`: `pandas` (ASCII, strict parsing/writing rules)
* `pcd`: `pypcd4`

## Definitions and quality gates

* Required geometry columns are `x`, `y`, `z` (lowercase in internal dataframe).
* For ASCII formats (`csv`, `xyz`), compare numeric fields with tolerance (`rtol=1e-6`, `atol=1e-9`).
* Preserve row count and core coordinates on read/write roundtrip.
* Unsupported extensions must fail with clear `ValueError`.

## Implementation phases

### Phase 1: Test scaffolding (keep pyntcloud in runtime)

* Add fixtures for new test data files:
	* `tests/testdata/xyz/diamond.xyz`
	* `tests/testdata/pcd_files/test_tree.pcd`
* Add tests for read behavior of `las`, `xyz`, `pcd` in `PointCloud.from_file`.
* Add roundtrip target tests for `xyz` and `pcd` writes as expected-fail markers until new writers are implemented.

Acceptance criteria:

* New fixtures resolve correctly.
* Read tests for `xyz` and `pcd` assert `x/y/z` and non-empty data.
* Target write tests exist and document intended behavior for Phase 2.

### Phase 2: Replace pyntcloud IO and remove dependency

* Replace file reader dispatch currently imported from `pyntcloud.io`.
* Implement native readers/writers for `xyz` and `pcd`.
* Keep current API behavior of `PointCloud.from_file` and `PointCloud.to_file`.
* Remove `PYNTCLOUD` instance conversion path and corresponding dependency code.
* Remove `pyntcloud` from `pyproject.toml` dependencies.

Acceptance criteria:

* All Phase 1 tests pass without editing test assertions.
* No runtime imports of `pyntcloud` remain.

### Phase 3: Documentation and examples

* Update README, Sphinx docs, and changelog.
* Remove pyntcloud references and add new backend notes (`laspy`, `pandas`, `pypcd4`).
* Verify notebooks and examples still execute with supported formats.

Acceptance criteria:

* No stale pyntcloud references in docs.
* Notebooks/examples run with updated dependencies.

### Phase 4: pandas 3 migration

* Raise pandas upper bound and run full test suite.
* Fix compatibility issues (dtype handling, IO parsing edge cases, indexing changes).

Acceptance criteria:

* Test suite passes with pandas 3.
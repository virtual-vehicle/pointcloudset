set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

# List available recipes
default:
    @just --list --unsorted

# Build HTML docs
[group('doc')]
doc:
    uv run --group doc sphinx-apidoc --no-toc --module-first -f -e -o ./doc/sphinx/source/python-api ./src/pointcloudset ./src/pointcloudset/config.py ./src/pointcloudset/io/dataset/commandline.py
    uv run --group doc --directory doc/sphinx sphinx-build -b html source build/html

# Check docstring coverage
[group('doc')]
doccoverage:
    uv run --group dev docstr-coverage src/pointcloudset --skip-magic --fail-under 68

# Run test suite
[group('qa')]
test:
    uv run --group dev pytest --cov=pointcloudset tests
    uv run --group dev pytest --cov-append --nbval-current-env --nbval-lax doc/sphinx/source/tutorial_notebooks/usage.ipynb
    uv run --group dev pytest --cov-append --nbval-current-env --nbval-lax doc/sphinx/source/tutorial_notebooks/reading_las_pcd.ipynb
    uv run --group dev pytest --cov-append --nbval-current-env --nbval-lax tests/notebooks/test_plot_plane.ipynb
    uv run --group dev pytest --cov-append --nbval-current-env --nbval-lax tests/notebooks/test_readme.ipynb
    uv run --group dev pytest --cov-append --nbval-current-env --nbval-lax tests/notebooks/test_animate.ipynb
    uv run --group dev python -m coverage report -i
    uv run --group dev python -m coverage html -i

# Lint with ruff
[group('qa')]
ruff:
    uv run --group dev ruff check src tests

# Auto-fix lint issues
[group('qa')]
ruff_fix:
    uv run --group dev ruff check src tests --fix



# Sort imports and forma[group('qa')]
[group('qa')]
ruff_format:
    uv run --group dev ruff check --select I --fix .
    uv run --group dev ruff format .

# Remove local artifacts
[group('qa')]
clean:
    find . -type f -name "*.pyc" -delete
    find . -type d -name "__pycache__" -exec rm -rf {} +
    rm -rf doc/sphinx/build doc/sphinx/source/python-api

# Bump patch release
[group('release')]
patch:
    uv version --bump patch
    git add pyproject.toml uv.lock CHANGELOG.rst
    git commit -m "bump version to $(uv version --short)"
    git tag "v$(uv version --short)"
    git push
    git push --tags

# Bump minor release
[group('release')]
minor:
    uv version --bump minor
    git add pyproject.toml uv.lock CHANGELOG.rst
    git commit -m "bump version to $(uv version --short)"
    git tag "v$(uv version --short)"
    git push
    git push --tags

# Build release image locally
[group('release')]
docker-local:
    docker build --rm -f docker/release.Dockerfile -t "tgoelles/pointcloudset:v$(uv version --short)" .
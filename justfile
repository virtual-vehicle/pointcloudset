set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

# List available recipes
default:
    @just --list --unsorted

# Build HTML docs
[group('doc')]
doc:
    uv run sphinx-apidoc --no-toc --module-first -f -e -o ./doc/sphinx/source/python-api ./src/pointcloudset ./src/pointcloudset/config.py ./src/pointcloudset/io/dataset/commandline.py
    uv run --directory doc/sphinx sphinx-build -b html source build/html

# Check docstring coverage
[group('doc')]
doccoverage:
    uv run docstr-coverage src/pointcloudset --skipmagic

# Run test suite
[group('qa')]
test:
    uv run pytest --cov=pointcloudset tests
    uv run pytest --cov-append --current-env --nbval-lax doc/sphinx/source/tutorial_notebooks/usage.ipynb
    uv run pytest --cov-append --current-env --nbval-lax doc/sphinx/source/tutorial_notebooks/reading_las_pcd.ipynb
    uv run pytest --cov-append --current-env --nbval-lax tests/notebooks/test_plot_plane.ipynb
    uv run pytest --cov-append --current-env --nbval-lax tests/notebooks/test_readme.ipynb
    uv run pytest --cov-append --current-env --nbval-lax tests/notebooks/test_animate.ipynb
    uv run python -m coverage report -i
    uv run python -m coverage html -i

# Lint with ruff
[group('qa')]
ruff:
    uv run ruff check src tests

# Auto-fix lint issues
[group('qa')]
ruff-fix:
    uv run ruff check src tests --fix

# Run mypy
[group('qa')]
mypy:
    uv run mypy -p pointcloudset --ignore-missing-imports

# Sort imports and format
[group('qa')]
sort-imports:
    uv run ruff check --select I --fix .
    uv run ruff format .

# Remove local artifacts
[group('qa')]
clean:
    py3clean .
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
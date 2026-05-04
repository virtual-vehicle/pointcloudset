default:
    @just --list

# Bump patch version, tag, and push
patch:
    uv version --bump patch
    git add pyproject.toml uv.lock CHANGELOG.rst
    git commit -m "bump version to $(uv version --short)"
    git tag "v$(uv version --short)"
    git push
    git push --tags

minor:
    uv version --bump minor
    git add pyproject.toml uv.lock CHANGELOG.rst
    git commit -m "bump version to $(uv version --short)"
    git tag "v$(uv version --short)"
    git push
    git push --tags
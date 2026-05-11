from __future__ import annotations

import importlib

import numpy as np


def njit(*_args, **_kwargs):
	def _decorator(func):
		return func

	return _decorator


_numba = None
try:
	_numba = importlib.import_module("numba")
except ImportError:
	HAS_NUMBA = False
else:
	HAS_NUMBA = True
	njit = _numba.njit


def _find_root_python(parent: np.ndarray, i: int) -> int:
	while parent[i] != i:
		parent[i] = parent[parent[i]]
		i = int(parent[i])
	return i


def _union_pairs_python(parent: np.ndarray, rank: np.ndarray, left: np.ndarray, right: np.ndarray) -> None:
	for idx in range(len(left)):
		root_a = _find_root_python(parent, int(left[idx]))
		root_b = _find_root_python(parent, int(right[idx]))
		if root_a == root_b:
			continue
		if rank[root_a] < rank[root_b]:
			parent[root_a] = root_b
		elif rank[root_a] > rank[root_b]:
			parent[root_b] = root_a
		else:
			parent[root_b] = root_a
			rank[root_a] += 1


def _roots_for_positions_python(parent: np.ndarray, positions: np.ndarray) -> np.ndarray:
	roots = np.empty(len(positions), dtype=np.intp)
	for idx in range(len(positions)):
		roots[idx] = _find_root_python(parent, int(positions[idx]))
	return roots


if HAS_NUMBA:

	@njit(cache=True)
	def _find_root_numba(parent: np.ndarray, i: int) -> int:
		while parent[i] != i:
			parent[i] = parent[parent[i]]
			i = parent[i]
		return i

	@njit(cache=True)
	def _union_pairs_numba(parent: np.ndarray, rank: np.ndarray, left: np.ndarray, right: np.ndarray) -> None:
		for idx in range(len(left)):
			root_a = _find_root_numba(parent, left[idx])
			root_b = _find_root_numba(parent, right[idx])
			if root_a == root_b:
				continue
			if rank[root_a] < rank[root_b]:
				parent[root_a] = root_b
			elif rank[root_a] > rank[root_b]:
				parent[root_b] = root_a
			else:
				parent[root_b] = root_a
				rank[root_a] += 1

	@njit(cache=True)
	def _roots_for_positions_numba(parent: np.ndarray, positions: np.ndarray) -> np.ndarray:
		roots = np.empty(len(positions), dtype=np.intp)
		for idx in range(len(positions)):
			roots[idx] = _find_root_numba(parent, positions[idx])
		return roots


def union_pairs(parent: np.ndarray, rank: np.ndarray, left: np.ndarray, right: np.ndarray) -> None:
	if HAS_NUMBA:
		_union_pairs_numba(parent, rank, left, right)
	else:
		_union_pairs_python(parent, rank, left, right)


def roots_for_positions(parent: np.ndarray, positions: np.ndarray) -> np.ndarray:
	if HAS_NUMBA:
		return _roots_for_positions_numba(parent, positions)
	return _roots_for_positions_python(parent, positions)

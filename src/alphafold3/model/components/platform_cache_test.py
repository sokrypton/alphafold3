"""the compilation cache must not be able to fill the disk"""
import os
import tempfile

from alphafold3.model.components.platform import prune_compilation_cache


def _cache(n=30, mb=1):
  d = tempfile.mkdtemp()
  for i in range(n):
    fp = os.path.join(d, f'e{i:02d}')
    with open(fp, 'wb') as fh:
      fh.write(b'\0' * (mb * 1024 * 1024))
    os.utime(fp, (1_000_000 + i, 1_000_000 + i))   # oldest access first
  return d


def test_pruning_evicts_oldest_until_under_budget():
  d = _cache()
  r = prune_compilation_cache(d, max_gb=10 / 1024)
  assert r['bytes'] <= 10 * 1024 ** 2
  # the recently USED shapes are the ones a session will want again
  assert sorted(os.listdir(d)) == [f'e{i:02d}' for i in range(20, 30)]


def test_pruning_under_budget_touches_nothing():
  d = _cache()
  assert prune_compilation_cache(d, max_gb=1.0)['removed'] == 0
  assert len(os.listdir(d)) == 30


def test_pruning_a_missing_cache_is_not_an_error():
  assert prune_compilation_cache('/nonexistent/cache/xyz', 1.0)['removed'] == 0

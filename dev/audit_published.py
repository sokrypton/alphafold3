"""Audit every blob before uploading it. RUN THIS BEFORE EVERY PUBLISH.

    PYTHONPATH=src:.:dev/oracles python dev/audit_published.py --apply

Two checks now, not four, and the file says why rather than pretending:

  records/sorted  -- write_params_blob sorts; an unsorted blob was NOT written
                     by it.
  applies         -- THE DECISIVE ONE: does the blob supply exactly what the
                     graph asks for, scope by scope and name by name?

The `missing[]` and `stamp` columns are gone with the shape manifests, which
were removed for never firing. What went with them was this file's ability to
run at all: it imported `converters.shapes`, so from that commit on it raised
ImportError before doing anything, and nobody noticed because nothing invoked
it.

**That cost five models.** On 2026-09-16 a user reported "the opendde weights
are broken", and opendde, boltz2, chai1, esmfold2 and esmfold2_fast all had
published blobs that predated a graph change -- opendde's single-cond LayerNorm
widened 831 -> 833, boltz2 gained cyclic conditioning, chai1 a structure single
projection, esmfold2's confidence re-embed 447 -> 451. Every one would have been
caught by `--apply`, which is the check that survived.

A blob that does not fit the graph raises at apply() time, or worse runs a head
on init values -- chai-1 shipped exactly that once.
"""
import argparse
import hashlib
import os
import sys


def _model_names():
  from alphafold3.model import model_registry
  return set(model_registry.MODEL_SPECS) | set(model_registry.ALIASES)


def audit(root, do_apply=False):
  global _MODEL_NAMES
  _MODEL_NAMES = _model_names()
  from converters.common import read_blob
  rows = []
  for m in sorted(os.listdir(root)):
    d = os.path.join(root, m)
    blob = os.path.join(d, '%s.bin.zst' % m)
    if not os.path.isdir(d) or not os.path.exists(blob):
      continue
    if m not in _MODEL_NAMES:
      # ESM-C is a weights artifact but NOT an AF3 model: it is the separate
      # graph ESMFold2 conditions on, with its own loader and its own blob
      # layout (one record per transformer block). It has no ModelSpec, no
      # canonical batch and no shape manifest, so every column here would be a
      # category error -- report it and leave it out of the verdict.
      rows.append(dict(model=m, records=None, sorted=None, missing=None,
                       stamp='not an AF3 model', applies='-',
                       sha=hashlib.sha256(open(blob, 'rb').read()).hexdigest()[:16]))
      continue
    recs = [(s, n) for s, n, _ in read_blob(blob) if s != '__meta__']
    row = dict(model=m, records=len(recs), sorted=recs == sorted(recs),
               sha=hashlib.sha256(open(blob, 'rb').read()).hexdigest()[:16])
    if do_apply:
      row['applies'] = _applies(m, d)
    rows.append(row)
  return rows


def _applies(model_name, model_dir):
  """True / 'FAILS: ...' -- does the blob supply exactly what the graph asks for?

  Deliberately NOT a full hk.transform(...).apply: a direct apply on a real batch
  raises a TracerArrayConversionError for EVERY model here (the batch carries
  non-array fields that AF3Runner.predict handles and a bare apply does not), so
  that check reports a property of the harness rather than of the blob. Comparing
  the init-traced parameter tree against the blob is the honest question, and it
  is what caught the since-removed protenix_tiny's absent template_embedding/z_norm.
  """
  # tokamax parses sys.argv lazily via absl on first import, and aborts on any
  # flag it does not know ("Unknown command line flag 'dir'"). Same guard as
  # convert.py: hide our argv from it.
  saved, sys.argv = sys.argv, sys.argv[:1]
  try:
    import haiku as hk
    import jax
    from alphafold3.model import model as af3_model, model_registry, params as afp
    from alphafold3.model.components import utils
    from converters import shapes
    cfg = af3_model.Model.Config()
    cfg.global_config.flash_attention_implementation = 'xla'
    model_registry.get(model_name).configure(cfg)
    # The batch used to come from the deleted `shapes.canonical_batch`.
    # fold_check builds the same thing and is what every gate already uses, so
    # this check now runs on the same featurisation the matrix does.
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'oracles'))
    import fold_check
    seq, _ = fold_check.parse_ca(os.path.expanduser('~/6MRR.pdb'))
    batch, cfg, _ = fold_check._fold_setup(model_name, seq,
                                           model_dir=model_dir)
    p = afp.get_model_haiku_params(model_dir=model_dir)
    want = jax.eval_shape(hk.transform(lambda b: af3_model.Model(cfg)(b)).init,
                          jax.random.PRNGKey(0),
                          utils.remove_invalidly_typed_feats(batch))
    w = {'%s/%s' % (s_, n) for s_, l in want.items() for n in l}
    h = {'%s/%s' % (s_, n) for s_, l in p.items() for n in l if s_ != '__meta__'}
    if w - h:
      return 'FAILS: %d wanted but absent, e.g. %s' % (len(w - h), sorted(w - h)[0][-70:])
    return True if not (h - w) else 'extra: %d in blob the graph never asks for' % len(h - w)
  except Exception as e:                      # noqa: BLE001 - report, don't raise
    return 'FAILS: %s' % str(e)[:90]
  finally:
    sys.argv = saved


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
  ap.add_argument('--dir', default='~/ported')
  ap.add_argument('--apply', action='store_true',
                  help='also try applying each blob to its graph (slow, decisive)')
  a = ap.parse_args(argv)
  rows = audit(os.path.expanduser(a.dir), a.apply)
  print('%-20s %-8s %-7s %-9s %s' % ('model', 'records', 'sorted', 'applies',
                                     'sha256'))
  for r in rows:
    if r['records'] is None:
      print('%-20s %-8s %-7s %-9s %s' % (r['model'], '-', '-', '-', r['sha']))
      continue
    ap_ = ('-' if not a.apply
           else ('yes' if r.get('applies') is True else 'NO'))
    print('%-20s %-8d %-7s %-9s %s' % (r['model'], r['records'], r['sorted'],
                                       ap_, r['sha']))
    if a.apply and r.get('applies') is not True:
      print('    apply -> %s' % r['applies'])
  # `applies` counts. A blob that does not fit the graph was being printed on a
  # side line and still passing: chai1 and protenix2 shipped a diffusion
  # conditioning at the wrong width for two commits because the verdict only
  # looked at the manifest, which is a converter-vs-blob check and cannot see
  # what the GRAPH asks for.
  bad = [r['model'] for r in rows
         if r['records'] is not None
         and (not r['sorted']
              or (a.apply and r.get('applies') is not True))]
  if not a.apply:
    print('\n(ran WITHOUT --apply, so the decisive check did not run)')
  print('\nNOT SAFE TO PUBLISH: %s' % (bad or 'none'))
  return 1 if bad else 0


if __name__ == '__main__':
  sys.exit(main())

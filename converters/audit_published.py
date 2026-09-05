"""Audit every published blob + shape manifest before re-uploading anything.

Four checks, because each has bitten us:

  records/sorted  -- write_params_blob sorts; an unsorted blob was NOT written by
                     it, so whatever the manifest says produced it did not.
  missing[]       -- parameters the GRAPH wants that the converter never made.
                     These are not cosmetic: the model raises at apply() time, or
                     worse, runs a head on init values (chai-1 shipped exactly
                     that in production).
  stamp           -- converter_sha256 vs the converter source as it stands now.
                     Note this hashes converters/<model>.py AND common.py, so a
                     change to common.py marks EVERY model stale at once.
  applies         -- the decisive one: can the shipped params actually be applied
                     to the graph they claim to be for?

Usage:  python -m converters.audit_published [--apply] [--dir ~/ported]
"""
import argparse
import hashlib
import json
import os
import sys


def audit(root, do_apply=False):
  from converters import shapes
  from converters.common import read_blob
  rows = []
  for m in sorted(os.listdir(root)):
    d = os.path.join(root, m)
    blob = os.path.join(d, '%s.bin.zst' % m)
    man = os.path.join(d, '%s.shapes.json' % m)
    if not os.path.isdir(d) or not os.path.exists(blob):
      continue
    recs = [(s, n) for s, n, _ in read_blob(blob) if s != '__meta__']
    row = dict(model=m, records=len(recs), sorted=recs == sorted(recs),
               sha=hashlib.sha256(open(blob, 'rb').read()).hexdigest()[:16])
    if os.path.exists(man):
      j = json.load(open(man))
      rec = (j.get('provenance') or {}).get('converter_sha256')
      row['stamp'] = 'current' if rec == shapes.provenance(m)['converter_sha256'] else 'STALE'
      row['missing'] = len(j.get('missing') or [])
    else:
      row['stamp'], row['missing'] = 'NO MANIFEST', -1
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
  is what caught protenix_tiny's absent template_embedding/z_norm.
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
    batch = shapes.canonical_batch(model_name, model_dir=model_dir)
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
  hdr = '%-20s %-8s %-7s %-8s %-9s %s' % ('model', 'records', 'sorted', 'missing',
                                          'stamp', 'sha256')
  print(hdr)
  for r in rows:
    print('%-20s %-8d %-7s %-8d %-9s %s' % (r['model'], r['records'], r['sorted'],
                                            r['missing'], r['stamp'], r['sha']))
    if a.apply and r.get('applies') is not True:
      print('    apply -> %s' % r['applies'])
  bad = [r['model'] for r in rows if r['missing'] > 0 or not r['sorted']]
  print('\nNOT SAFE TO PUBLISH: %s' % (bad or 'none'))
  return 1 if bad else 0


if __name__ == '__main__':
  sys.exit(main())

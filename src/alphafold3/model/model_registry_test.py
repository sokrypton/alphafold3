# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Tests for the per-model settings registry."""

from absl.testing import absltest
from absl.testing import parameterized
from alphafold3.model import model as af3_model
from alphafold3.model import model_config
from alphafold3.model import model_registry


class RegistryTest(parameterized.TestCase):

  def test_every_model_the_graph_knows_has_a_spec(self):
    # model_config.MODELS is what global_config.model may be set to; a name the
    # forward branches key on but the registry cannot configure would build a
    # graph at the wrong shapes.
    self.assertCountEqual(model_config.MODELS, model_registry.MODEL_SPECS)

  def test_aliases_resolve_to_real_models(self):
    for alias, name in model_registry.ALIASES.items():
      with self.subTest(alias=alias):
        self.assertEqual(model_registry.get(alias).name, name)

  def test_unknown_model_is_rejected(self):
    with self.assertRaises(ValueError):
      model_registry.get('alphafold4')

  def test_configuring_alphafold3_changes_nothing(self):
    """Stock AF3 must be byte-identical to the config with no registry at all.

    Every ported family is a branch off this one, so the moment configuring
    'alphafold3' moves a single channel, every claim about not disturbing
    AlphaFold 3 stops being true.
    """
    untouched = af3_model.Model.Config().as_dict()
    configured = model_registry.get('alphafold3').configure(
        af3_model.Model.Config()).as_dict()
    configured['global_config'].pop('model', None)
    untouched['global_config'].pop('model', None)
    self.assertEqual(untouched, configured)

  def test_model_name_lands_in_global_config(self):
    for name in model_registry.MODEL_SPECS:
      with self.subTest(model=name):
        config = model_registry.get(name).configure(af3_model.Model.Config())
        self.assertEqual(config.global_config.model, name)

  @parameterized.parameters(
      ('intellifold2', 512, 256),
      ('opendde', 384, 128),
      ('protenix2', 256, 128),
      ('chai1', 256, 64),
      ('boltz2', 128, 64),
      ('openfold3', 128, 64),
      ('rosettafold3', 128, 64),
      ('alphafold3', 128, 64),
  )
  def test_pair_and_msa_channels(self, name, pair_channel, msa_channel):
    config = model_registry.get(name).configure(af3_model.Model.Config())
    self.assertEqual(config.evoformer.pair_channel, pair_channel)
    self.assertEqual(config.evoformer.msa_channel, msa_channel)

  def test_only_alphafold3_uses_af3s_hardcoded_fourier_embedding(self):
    """Every ported family carries a TRAINED Fourier noise embedding.

    It is not derivable from OPENFOLD3_LINEAGE alone: chai-1 and IntelliFold-2
    both have one and neither is in that tuple, and falling back to AF3's
    hardcoded constants is silent.
    """
    trained = {n for n in model_registry.MODEL_SPECS
               if model_registry.get(n).trained_fourier}
    self.assertEqual(trained,
                     set(model_registry.MODEL_SPECS) - {'alphafold3'})

  def test_boltz2_brings_its_own_sampler_constants(self):
    # AF3's EDM constants are not universal; running boltz2's network on them is
    # silent (nothing errors, the sampler just anneals on the wrong schedule).
    config = model_registry.get('boltz2').configure(af3_model.Model.Config())
    default = af3_model.Model.Config()
    self.assertNotEqual(config.heads.diffusion.eval.gamma_0,
                        default.heads.diffusion.eval.gamma_0)
    self.assertAlmostEqual(config.heads.diffusion.eval.gamma_0, 0.605)

  def test_protenix_variants_appear_wherever_protenix2_does(self):
    """A Protenix model type takes every forward branch protenix2 takes.

    They differ only in counts and widths -- block counts and c_z, both read off
    the checkpoint by converters/protenix2.derive_dims -- so any branch keyed on
    'protenix2' is keyed on the wrong thing unless mini and tiny are beside it.

    This is a test rather than a convention because the failure mode is bad:
    while porting mini, three separate lists were missed, and each surfaced only
    as a shape error at load or a count of uncovered parameters. Nothing said
    "you forgot a model in a tuple". Every membership list lives in model_config
    precisely so this test can see all of them.
    """
    lists = {name: value for name, value in vars(model_config).items()
             if name.isupper() and isinstance(value, tuple)
             and all(isinstance(v, str) for v in value)}
    checked = 0
    for name, members in lists.items():
      if 'protenix2' not in members:
        continue
      checked += 1
      for variant in model_config.PROTENIX_FAMILY:
        self.assertIn(
            variant, members,
            f'{name} contains protenix2 but not {variant}; every member of '
            'model_config.PROTENIX_FAMILY takes the same forward branches')
    self.assertGreater(checked, 0, 'no protenix2 membership lists found')

  def test_padded_key_windows_imply_the_or_mask(self):
    """A model that PADS its atom key window must mask keys from real queries.

    AF3's atom cross-attention biases with (mask_q - 1) * (mask_k - 1), an AND
    that penalises a pair only when both ends are invalid. That is safe only
    because AtomCrossAtt slides an out-of-bounds window back in bounds, so no key
    is ever padding. The moment a family pads instead, every padded key becomes
    fully attendable from every real query -- silently, and only in the edge
    windows, so no RMSD gate sees it. protenix2 and opendde shipped that way.

    KEY_MASKED_ATOM_ATTENTION is deliberately a SUPERSET (rosettafold3 is in it
    without the padded_keys knob, because it reaches the same place through a
    short atom count rather than through padding), so this asserts containment,
    not equality.
    """
    padded = {n for n in model_registry.MODEL_SPECS
              if model_registry.get(n).featurise.get('padded_keys')}
    self.assertContainsSubset(padded,
                              model_config.KEY_MASKED_ATOM_ATTENTION)

  def test_output_terms_never_claim_a_licence_we_have_not_established(self):
    for name in model_registry.MODEL_SPECS:
      spec = model_registry.get(name)
      with self.subTest(model=name):
        terms = spec.output_terms()
        self.assertIn('TERMS OF USE', terms)
        if name != 'alphafold3' and spec.weights_licence is None:
          self.assertIn('we have not\nestablished', terms)


if __name__ == '__main__':
  absltest.main()

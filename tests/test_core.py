import unittest
import numpy as np
import torch
from fpp_net import FPPNet,spectral_features

class CoreTests(unittest.TestCase):
 def test_common_gain_invariance(self):
  rng=np.random.default_rng(42);x=rng.normal(size=(3,3,1710)).astype(np.float64)
  np.testing.assert_allclose(spectral_features(x,10000),spectral_features(x*7,10000),atol=2e-6,rtol=2e-6)
 def test_dimensions_parameters_and_gradient(self):
  for c,count in [(1,17210),(3,17434)]:
   m=FPPNet(c);self.assertEqual(sum(p.numel() for p in m.parameters()),count)
   x=torch.randn(2,2*c,1024);y=m(x);self.assertEqual(tuple(y.shape),(2,2));y.square().mean().backward()
   self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters()))
 def test_reject_nonfinite_raw(self):
  with self.assertRaises(ValueError):spectral_features(np.full((1,1,5120),np.nan),25600)
if __name__=='__main__':unittest.main()

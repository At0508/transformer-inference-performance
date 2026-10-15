import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "optional PyTorch dependency is not installed")
class CacheEquivalenceTests(unittest.TestCase):
    def test_incremental_decode_matches_full_prefill(self):
        from inference_lab.torch_model import TinyDecoder

        torch.manual_seed(1)
        model = TinyDecoder(vocabulary_size=64, hidden_size=16, heads=4,
                            layers=2, max_sequence_length=16,
                            feed_forward_size=32).eval()
        tokens = torch.tensor([[1, 3, 5, 7, 9]])
        with torch.inference_mode():
            full_logits, _ = model.prefill(tokens)
            _, cache = model.prefill(tokens[:, :-1])
            incremental_logits, new_cache = model.decode_step(tokens[:, -1], cache)
        self.assertTrue(torch.allclose(full_logits[:, -1], incremental_logits,
                                       atol=1e-5, rtol=1e-5))
        self.assertEqual(new_cache[0][0].shape[2], tokens.shape[1])


if __name__ == "__main__":
    unittest.main()

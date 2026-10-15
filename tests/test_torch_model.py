import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "optional PyTorch dependency is not installed")
class CacheEquivalenceTests(unittest.TestCase):
    def test_multiple_cached_steps_match_full_prefix_for_two_sequences(self):
        from inference_lab.cost_model import ModelShape
        from inference_lab.torch_model import TinyDecoder
        from inference_lab.benchmark import generate_cached, generate_uncached, _benchmark_case

        torch.manual_seed(3)
        torch.set_num_threads(1)
        model = TinyDecoder(64, 16, 4, 2, 16, 32).eval()
        shape = ModelShape(layers=2, hidden_size=16, attention_heads=4,
                           feed_forward_size=32, vocabulary_size=64, max_sequence_length=16)
        self.assertEqual(shape.parameter_count, sum(p.numel() for p in model.parameters()))
        tokens = torch.randint(0, 64, (2, 7))
        with torch.inference_mode():
            _, cache = model.prefill(tokens[:, :3])
            for position in range(3, 7):
                incremental, cache = model.decode_step(tokens[:, position], cache)
                full, _ = model.prefill(tokens[:, :position + 1])
                torch.testing.assert_close(incremental, full[:, -1], atol=1e-5, rtol=1e-5)
                self.assertEqual(cache[0][0].shape[2], position + 1)
            logits, base_cache = model.prefill(tokens[:, :3])
            first = logits[:, -1].argmax(-1)
            original_length = base_cache[0][0].shape[2]
            torch.testing.assert_close(generate_cached(model, first, base_cache, 3),
                                       generate_uncached(model, tokens[:, :3], first, 3))
            self.assertEqual(base_cache[0][0].shape[2], original_length)
        report = _benchmark_case(model, torch.device('cpu'), 2, 3, 4, 1, 3, 4)
        self.assertTrue(report['greedy_outputs_match'])
        self.assertEqual(len(report['decode_ms_samples_total']), 3)
        self.assertAlmostEqual(report['decode_tokens_per_second'],
                               1000 * 2 * 3 / report['cached_decode_ms_median'])
        self.assertEqual(report['kv_cache_bytes_after_prefill_actual'], 2 * 2 * 2 * 3 * 16 * 4)

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

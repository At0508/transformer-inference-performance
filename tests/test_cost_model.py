import unittest

from inference_lab.cost_model import ModelShape, estimate


class CostModelTests(unittest.TestCase):
    def test_kv_cache_formula_and_scaling(self):
        shape = ModelShape(layers=2, hidden_size=32, attention_heads=4,
                           feed_forward_size=64, vocabulary_size=128)
        one = estimate(shape, batch_size=1, prompt_tokens=8, generated_tokens=2,
                       bytes_per_element=2)
        two = estimate(shape, batch_size=2, prompt_tokens=8, generated_tokens=2,
                       bytes_per_element=2)
        self.assertEqual(one.kv_cache_bytes_at_end, 2 * 2 * 1 * 10 * 4 * 8 * 2)
        self.assertEqual(two.kv_cache_bytes_at_end, 2 * one.kv_cache_bytes_at_end)

    def test_roofline_bound_is_positive_and_optional(self):
        shape = ModelShape(layers=1, hidden_size=16, attention_heads=4,
                           feed_forward_size=32, vocabulary_size=64)
        estimate_row = estimate(shape, batch_size=1, prompt_tokens=4,
                                generated_tokens=2, peak_flops_per_second=1e12,
                                memory_bandwidth_bytes_per_second=1e11)
        self.assertGreater(estimate_row.ideal_prefill_ms, 0)
        self.assertGreater(estimate_row.ideal_decode_ms, 0)
        no_roofline = estimate(shape, batch_size=1, prompt_tokens=4,
                               generated_tokens=2)
        self.assertIsNone(no_roofline.ideal_prefill_ms)

    def test_rejects_invalid_shape_or_workload(self):
        with self.assertRaises(ValueError):
            ModelShape(hidden_size=31, attention_heads=4)
        with self.assertRaises(ValueError):
            estimate(ModelShape(), batch_size=0, prompt_tokens=8,
                     generated_tokens=2)


if __name__ == "__main__":
    unittest.main()

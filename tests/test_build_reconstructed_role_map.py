"""Small deterministic checks; no model weights or datasets are needed."""

import copy
import unittest

from scripts.build_reconstructed_role_map import (
    build_role_map,
    compare_maps,
    linear_quantile,
    validate_counts,
)


def fixture():
    # Two clients share domain A; the token-weighted A frequency is [4/6, 2/6, 0].
    return {
        "schema": "role-router-counts-v1",
        "model": "synthetic-model",
        "layer_specs": {"0": {"num_experts": 3, "top_k": 1}},
        "clients": [
            {
                "client_id": "a0", "domain": "A", "n_valid_tokens": 4,
                "layers": {"0": [4, 0, 0]},
            },
            {
                "client_id": "a1", "domain": "A", "n_valid_tokens": 2,
                "layers": {"0": [0, 2, 0]},
            },
            {
                "client_id": "b0", "domain": "B", "n_valid_tokens": 3,
                "layers": {"0": [0, 3, 0]},
            },
        ],
    }


class RoleMapTests(unittest.TestCase):
    def test_domain_weighting_and_role_precedence(self):
        roles = build_role_map(
            fixture(), mode="explicit", tau_red=0.0, tau_g=0.3, tau_h=0.5,
        )["layers"]["0"]
        self.assertEqual(roles["0"], {"role": "adaptable", "active_domains": ["A"]})
        self.assertEqual(roles["1"], {"role": "shared", "active_domains": ["A", "B"]})
        self.assertEqual(roles["2"], {"role": "redundant", "active_domains": []})

    def test_redundant_clears_domain_membership_for_existing_pruner(self):
        roles = build_role_map(
            fixture(), mode="explicit", tau_red=0.4, tau_g=0.6, tau_h=0.5,
        )["layers"]["0"]
        self.assertEqual(roles["0"], {"role": "redundant", "active_domains": []})

    def test_invalid_top_k_sum_rejected(self):
        counts = fixture()
        counts["clients"][0]["layers"]["0"] = [3, 0, 0]
        with self.assertRaisesRegex(ValueError, "expected n_valid_tokens"):
            validate_counts(counts)

    def test_missing_layer_rejected(self):
        counts = fixture()
        counts["clients"][0]["layers"] = {}
        with self.assertRaisesRegex(ValueError, "cover exactly"):
            validate_counts(counts)

    def test_requires_two_domains_and_explicit_cutoffs(self):
        counts = fixture()
        counts["clients"] = counts["clients"][:2]
        with self.assertRaisesRegex(ValueError, "two distinct domains"):
            validate_counts(counts)
        with self.assertRaisesRegex(ValueError, "requires both"):
            build_role_map(fixture(), mode="explicit", tau_h=0.8)

    def test_quantile_is_linear_and_reference_check_detects_difference(self):
        self.assertEqual(linear_quantile([0.0, 1.0], 0.25), 0.25)
        candidate = build_role_map(fixture(), mode="qwen-quantile", tau_h=0.75)
        reference = copy.deepcopy(candidate)
        self.assertEqual(compare_maps(candidate, reference), [])
        reference["layers"]["0"]["1"]["role"] = "redundant"
        self.assertEqual(len(compare_maps(candidate, reference)), 1)


if __name__ == "__main__":
    unittest.main()

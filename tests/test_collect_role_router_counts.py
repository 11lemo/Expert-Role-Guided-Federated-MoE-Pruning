"""Lightweight collector checks that do not require a model or PyTorch."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.collect_role_router_counts import discover_moe_gates, select_client_texts


class ClientSelectionTests(unittest.TestCase):
    def test_repeat_and_split_like_domain_loader(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            domain_dir = Path(temporary)
            (domain_dir / "squad.json").write_text(
                json.dumps(["a long valid sample", "another valid sample", "short"]),
                encoding="utf-8",
            )
            first = select_client_texts(domain_dir, ["squad"], 3, 2, 0)
            again = select_client_texts(domain_dir, ["squad"], 3, 2, 0)
            self.assertEqual(first, again)
            self.assertEqual([item["client_id"] for item in first], ["squad_0", "squad_1"])
            self.assertEqual([len(item["texts"]) for item in first], [3, 3])
            self.assertEqual(set(first[0]["texts"] + first[1]["texts"]),
                             {"a long valid sample", "another valid sample"})


class GateDiscoveryTests(unittest.TestCase):
    def test_sparse_layer_retains_decoder_index(self):
        gate = SimpleNamespace(out_features=3)
        sparse_mlp = SimpleNamespace(gate=gate, experts=[object()] * 3, num_experts=3, top_k=2)
        layers = [SimpleNamespace(mlp=SimpleNamespace()), SimpleNamespace(mlp=sparse_mlp)]
        model = SimpleNamespace(
            config=SimpleNamespace(model_type="qwen2_moe"),
            model=SimpleNamespace(layers=layers),
        )
        backbone, gates, specs = discover_moe_gates(model)
        self.assertIs(backbone, model.model)
        self.assertEqual(gates, {"1": gate})
        self.assertEqual(specs, {"1": {"num_experts": 3, "top_k": 2}})

    def test_reject_unsupported_model(self):
        model = SimpleNamespace(config=SimpleNamespace(model_type="llama"))
        with self.assertRaises(ValueError):
            discover_moe_gates(model)


if __name__ == "__main__":
    unittest.main()

"""Offline round-trip checks for the current three-head model (not paper metrics)."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
HAS_ML = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "safetensors"))


@unittest.skipUnless(HAS_ML, "optional neural dependencies are not installed")
class PaperCheckpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except (ImportError, OSError) as exc:
            raise unittest.SkipTest(f"optional neural runtime cannot load: {type(exc).__name__}") from exc

    def test_all_variants_round_trip_without_network(self):
        import torch
        from transformers import BertConfig, BertTokenizerFast
        from run_macbert_mgp_module_ablation import PAPER_FIELDS, VARIANTS, create_model, save_checkpoint
        from predict_macbert_mgp import build_inference_rows, load_checkpoint, predict_rows

        torch.set_num_threads(1)
        maps = {field: {"label_a": 0, "label_b": 1} for field in PAPER_FIELDS}
        maps["relations"] = {"HAS_FAILURE_MODE": 0}
        weights = {field: [1.0, 2.0] for field in PAPER_FIELDS}
        rows = build_inference_rows([{"document_id": "synthetic_1", "text": "water pump leak",
                                     "chains": [{"component_id": "must_not_be_used"}]}])
        self.assertEqual(rows[0]["chains"], [])
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            vocab = folder / "vocab.txt"
            vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nwater\npump\nleak\n", encoding="utf-8")
            tokenizer = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=True)
            config = BertConfig(vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1,
                                num_attention_heads=2, intermediate_size=32, max_position_embeddings=32)
            for variant, flags in VARIANTS.items():
                with self.subTest(variant=variant):
                    torch.manual_seed(42)
                    model = create_model(None, maps, weights, flags, torch, torch.nn, encoder_config=config)
                    target = save_checkpoint(folder / variant, model, tokenizer, maps, weights, flags,
                                             max_length=16, variant=variant, seed=42,
                                             best_epoch=1, best_validation_loss=1.25)
                    loaded, loaded_tokenizer, metadata = load_checkpoint(target, "cpu")
                    self.assertEqual(metadata["checkpoint_selection"], "minimum_validation_loss")
                    self.assertEqual(metadata["class_threshold"], 0.5)
                    for key, value in model.state_dict().items():
                        self.assertTrue(torch.equal(value, loaded.state_dict()[key]), key)
                    before = predict_rows(model, tokenizer, metadata, rows, device="cpu")
                    after = predict_rows(loaded, loaded_tokenizer, metadata, rows, device="cpu")
                    self.assertEqual(before, after)
                    self.assertEqual(set(after[0]["predicted"]), set(PAPER_FIELDS))

    def test_inference_rejects_duplicate_ids(self):
        from predict_macbert_mgp import build_inference_rows
        with self.assertRaisesRegex(ValueError, "Duplicate document ID"):
            build_inference_rows([{"document_id": "duplicate", "text": "a"},
                                  {"document_id": "duplicate", "text": "b"}])


if __name__ == "__main__":
    unittest.main()

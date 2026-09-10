from __future__ import annotations

from pathlib import Path

import torch


def export_onnx(
    model: torch.nn.Module,
    output_path: str | Path,
    vocab_size: int,
    max_len: int = 128,
    max_word_len: int = 20,
    opset: int = 17,
) -> Path:
    """Export encoder+classifier emissions for edge runtimes (CRF decoded separately)."""
    import torch.onnx

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy_ids = torch.zeros(1, max_len, max_word_len, dtype=torch.long)
    dummy_mask = torch.ones(1, max_len, dtype=torch.bool)

    class EmissionWrapper(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return self.inner.classifier(self.inner.encode(input_ids, mask))

    wrapper = EmissionWrapper(model)
    wrapper.eval()
    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_mask),
        str(output_path),
        input_names=["input_ids", "mask"],
        output_names=["emissions"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq", 2: "word_chars"},
            "mask": {0: "batch", 1: "seq"},
            "emissions": {0: "batch", 1: "seq"},
        },
        opset_version=opset,
        dynamo=False,
    )
    return output_path


def export_espdl_stub(checkpoint_path: str, output_dir: str) -> None:
    """Placeholder for ESP-PPQ quantization — run on Linux/GPU box with esp-ppq installed."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    readme = out / "README_ESP_PPQ.txt"
    readme.write_text(
        "ESP-PPQ export is not run on this machine yet.\n"
        "After SSH to GPU/Linux host:\n"
        "  1. pip install esp-ppq\n"
        "  2. Export ONNX via scripts/export_model.py\n"
        "  3. Run espdl_quantize_onnx / espdl_quantize_torch per ESP-DL docs\n"
        f"Checkpoint: {checkpoint_path}\n",
        encoding="utf-8",
    )

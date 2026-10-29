"""
Track 2 ONNX Export Entry Point for Team SAT (MIT-BLR).
Regenerates submitted_model.onnx from weights/best.pt for Submission #19447.
"""
import os
import torch
import onnx
from src.track2_apex_orin import Track2ApexOrin

RES = (1024, 1024)
NUM_CLASSES = 5


def main(model_dir, out_onnx):
    ckpt_path = os.path.join(model_dir, "best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    model = Track2ApexOrin(pretrained_backbone=False)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    model.eval()

    os.makedirs(os.path.dirname(out_onnx) or ".", exist_ok=True)
    H, W = RES
    dummy_x = torch.randn(1, 3, H, W, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_x,
        out_onnx,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["restored_image", "seg_logits"],
        dynamic_axes=None,
        dynamo=False
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="weights")
    parser.add_argument("--out-onnx", default="onnx/submitted_model.onnx")
    args = parser.parse_args()
    main(args.model_dir, args.out_onnx)

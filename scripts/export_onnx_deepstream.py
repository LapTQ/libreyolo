import os
import argparse
import torch
import torch.nn as nn
from libreyolo import LibreYOLO

class DeepStreamModelWrapper(nn.Module):
    """
    Wraps a LibreYOLO model to produce a consolidated DeepStream-compatible output.
    
    The wrapped model transposes the output format from:
        (batch, 4 + nc, total_anchors)
    to:
        (batch, total_anchors, 6) containing [x1, y1, x2, y2, score, class_id]
        
    This matches the custom bounding box parsing layer layout expected by 
    the DeepStream YOLO parser.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, tuple):
            # For segmentation or dual heads, the first element is usually the detection predictions
            # Shape of predictions: (batch, 4 + nc, total_anchors)
            predictions = out[0]
        else:
            predictions = out
            
        # Transpose to (batch, total_anchors, 4 + nc)
        predictions = predictions.transpose(1, 2)
        
        # Extract bounding boxes: (batch, total_anchors, 4)
        boxes = predictions[:, :, :4]
        
        # Get maximum confidence scores and class indices: (batch, total_anchors, 1)
        scores, labels = torch.max(predictions[:, :, 4:], dim=-1, keepdim=True)
        
        # Concatenate: (batch, total_anchors, 6)
        return torch.cat([boxes, scores, labels.to(boxes.dtype)], dim=-1)

def main():
    parser = argparse.ArgumentParser(description="LibreYOLO DeepStream ONNX Export")
    parser.add_argument("-w", "--weights", required=True, type=str, help="Input weights (.pt) file path")
    parser.add_argument("-s", "--size", nargs="+", type=int, default=[640], help="Inference size [H,W] or [size]")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--simplify", action="store_true", help="ONNX simplify model")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic batch-size")
    parser.add_argument("--batch", type=int, default=1, help="Static batch-size")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    if not os.path.isfile(args.weights):
        raise RuntimeError(f"Invalid weights file: {args.weights}")
    
    device = torch.device(args.device)
    
    print(f"Loading LibreYOLO model from {args.weights} on {device}...")
    model = LibreYOLO(args.weights, device=args.device)
    
    # Fuse Conv+BN and RepConvN layers for faster/cleaner export
    print("Fusing model layers...")
    model.model.fuse()
    model.model.eval()

    # Set up export flags on modules
    for m in model.model.modules():
        t = type(m)
        if t is nn.Upsample and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None
            
    head = model.model.head
    if hasattr(head, "inplace"):
        head.inplace = False
    head.dynamic = args.dynamic
    head.export = True

    # Wrap the model for DeepStream output format
    print("Wrapping model for DeepStream output...")
    ds_model = DeepStreamModelWrapper(model.model)
    ds_model.eval()

    # Output labels file
    if hasattr(model, "names") and model.names:
        labels_file = os.path.join(os.path.dirname(args.weights) or ".", "labels.txt")
        print(f"Creating labels file: {labels_file}")
        with open(labels_file, "w", encoding="utf-8") as f:
            for name in model.names.values():
                f.write(f"{name}\n")

    # Define resolution
    img_size = args.size * 2 if len(args.size) == 1 else args.size
    print(f"Export resolution: {img_size[0]}x{img_size[1]}")
    
    dummy_input = torch.zeros(args.batch, 3, *img_size).to(device)
    
    onnx_output_file = args.weights.rsplit(".", 1)[0] + ".onnx"
    
    dynamic_axes = {
        "input": {0: "batch"},
        "output": {0: "batch"}
    }
    
    print(f"Exporting ONNX model to {onnx_output_file}...")
    torch.onnx.export(
        ds_model,
        dummy_input,
        onnx_output_file,
        verbose=False,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes if args.dynamic else None
    )

    if args.simplify:
        print("Simplifying ONNX model...")
        try:
            import onnx
            import onnxslim
            model_onnx = onnx.load(onnx_output_file)
            model_onnx = onnxslim.slim(model_onnx)
            onnx.save(model_onnx, onnx_output_file)
            print("Simplification complete.")
        except ImportError:
            print("onnxslim or onnx package not found. Skipping simplification.")

    print(f"Export successful: {onnx_output_file}")

if __name__ == "__main__":
    main()

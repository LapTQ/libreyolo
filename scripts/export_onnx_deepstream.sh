python3 scripts/export_onnx_deepstream.py \
    -w outputs/train/v1.person.LibreYOLO9s/weights/best.pt \
    --simplify \
    --size 640 \
    --dynamic \
    # --batch 5
    # --opset 17 \
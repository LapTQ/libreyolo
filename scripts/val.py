from libreyolo import LibreYOLO

# Fine-tune from a pretrained checkpoint (recommended)
model = LibreYOLO("models/checkpoints/fs26/detection/libreyolo/v6.person.LibreYOLO9t/weights/best.pt", device="1")

results = model.val(
    data="libreyolo/config/fs26/v1.person.yaml",  # dataset config
    batch=16,
    imgsz=640,
    conf=0.001,  # low conf for mAP calculation
    iou=0.6,  # NMS IoU threshold
    # device="1",
    workers=8,
    split="val",  # "val", "test", or "train"
    save_json=False,  # save predictions as COCO JSON
    verbose=True,  # print per-class metrics
)

print(f"mAP50:    {results['metrics/mAP50']:.3f}")
print(f"mAP50-95: {results['metrics/mAP50-95']:.3f}")
print(f"precision:    {results['metrics/precision']:.3f}")
print(f"recall: {results['metrics/recall']:.3f}")

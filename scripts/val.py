from libreyolo import LibreYOLO

# Fine-tune from a pretrained checkpoint (recommended)
model = LibreYOLO("outputs/train/v1.person.LibreYOLO9s/weights/best.pt")

results = model.val(
    data="libreyolo/config/datasets/fs26.v1.person.yaml",  # dataset config
    batch=32,
    imgsz=640,
    conf=0.001,  # low conf for mAP calculation
    iou=0.6,  # NMS IoU threshold
    device="0",
    workers=8,
    split="val",  # "val", "test", or "train"
    save_json=False,  # save predictions as COCO JSON
    verbose=True,  # print per-class metrics
)

print(f"mAP50:    {results['metrics/mAP50']:.3f}")
print(f"mAP50-95: {results['metrics/mAP50-95']:.3f}")
print(f"precision:    {results['metrics/precision']:.3f}")
print(f"recall: {results['metrics/recall']:.3f}")

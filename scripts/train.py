from libreyolo import LibreYOLO

# Fine-tune from a pretrained checkpoint (recommended)
model = LibreYOLO("LibreYOLO9s.pt")

results = model.train(
    data="libreyolo/config/datasets/fs26.v1.person.yaml",     # path to data.yaml (required)

    # Schedule
    epochs=100,              # default: 300
    batch=128,
    imgsz=640,

    # Optimizer
    lr0=0.01,                # initial learning rate
    optimizer="SGD",         # "SGD", "Adam", "AdamW"

    # System
    device="",              # "" | "cpu" | "cuda" | "0" | "0,1"
    workers=8,
    seed=0,

    # Output
    project="outputs/train",
    name="v1.person.LibreYOLO9s",
    exist_ok=False,

    # Training features
    amp=True,                # automatic mixed precision
    patience=30,             # early stopping patience
    resume=False,            # resume from loaded checkpoint

    eval_interval=1,
)

print(f"Best mAP50-95: {results['best_mAP50_95']:.3f}")
print(f"Best mAP50: {results['best_mAP50']:.3f}")
print(f"Best epoch: {results['best_epoch']}")
print(f"Best checkpoint: {results['best_checkpoint']}")
# If needed:
# !pip install -U ultralytics

from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import sys
import traceback

from ultralytics import YOLO

# --- Your base config ---
DATA_YAML = "dd3d/data.yaml"
PROJECT_ROOT = "runs15"         # all experiments go here
EPOCHS = 15
IMGSZ = 896

# Choose your two models here (the assignment says "both models")
MODELS = [
    "yolo11n.pt",
]

# Sweeps required by assignment
LR_MULTS = [1.0, 5.0, 0.2]     # default, 5x, 0.2x
BATCHES = []         # batch sweep

try:
    import torch
except ImportError:
    torch = None

def run_name(model_ckpt: str, imgsz: int, batch: int, lr_mult: float) -> str:
    base = Path(model_ckpt).stem  # "yolo11s"
    lr_tag = f"lr{lr_mult:g}x"     # lr1x, lr5x, lr0.2x
    return f"{base}_img{imgsz}_b{batch}_{lr_tag}"

def run_dir(project: str, name: str) -> Path:
    return Path(project) / name

def load_results_csv(project: str, name: str) -> pd.DataFrame:
    csv_path = run_dir(project, name) / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing results.csv at: {csv_path}")
    df = pd.read_csv(csv_path)
    # YOLO logs epochs starting at 0 in the csv typically; we'll create a 1-based epoch column for plotting
    if "epoch" in df.columns:
        df["epoch_1based"] = df["epoch"] + 1
    else:
        # fallback: assume row index corresponds to epoch
        df["epoch_1based"] = df.index + 1
    return df


def pick_col(df: pd.DataFrame, candidates):
    """Return the first existing column name from candidates, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def plot_metric(df: pd.DataFrame, y_col: str, title: str, label: str = None, 
                x_col: str = "epoch_1based"):
    plt.plot(df[x_col], df[y_col], marker="o", linewidth=1.5, label=label)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.grid(True, alpha=0.3)

def finalize_plot(ylabel: str, legend=True):
    plt.ylabel(ylabel)
    if legend:
        plt.legend()
    plt.tight_layout()
    plt.show()

def is_oom_error(exc: Exception) -> bool:
    text = str(exc).lower()
    markers = [
        "out of memory",
        "cuda out of memory",
        "cudnn_status_alloc_failed",
        "cuda error: out of memory",
    ]
    return any(m in text for m in markers)


def train_one(model_ckpt: str, imgsz: int, batch: int, lr_mult: float, project: str, epochs: int):
    name = run_name(model_ckpt, imgsz, batch, lr_mult)

    # Skip if already trained (optional)
    out_dir = run_dir(project, name)
    if (out_dir / "results.csv").exists():
        print(f"[SKIP] {name} already exists.")
        return name

    # Ultralytics hyperparams:
    # - lr0 is initial learning rate; default depends on model/task, so we get "default ×" by not setting it.
    # - BUT for 5× and 0.2× we need a baseline. The cleanest way is:
    #   1) run default with lr_mult=1.0 (no lr0 override)
    #   2) for others, override lr0 relative to a chosen baseline value you define
    #
    # If your class expects strict "× default", then you should set BASE_LR0 to the default used in your training config.
    # Common default for YOLO detect is around 0.01, but verify in your training logs.
    BASE_LR0 = 0.01

    requested_batch = batch
    fallback_batches = [requested_batch]
    if requested_batch > 1:
        fallback_batches.append(max(1, requested_batch // 2))
    if requested_batch > 2:
        fallback_batches.append(max(1, requested_batch // 4))
    # de-dup while preserving order
    fallback_batches = list(dict.fromkeys(fallback_batches))

    last_exc = None
    for idx, try_batch in enumerate(fallback_batches, start=1):
        model = YOLO(model_ckpt)
        train_kwargs = dict(
            data=DATA_YAML,
            epochs=epochs,
            imgsz=imgsz,
            batch=try_batch,
            project=project,
            name=name,
            lr0=BASE_LR0 * lr_mult,  # Set learning rate directly
            # device=0,  # uncomment if you want GPU index
            plots=True,  # YOLO will generate built-in plots (results, PR, F1, confusion, etc.)
            verbose=True,
        )

        try:
            if try_batch != requested_batch:
                print(
                    f"[RETRY] {name}: trying smaller batch={try_batch} "
                    f"(attempt {idx}/{len(fallback_batches)})"
                )
            else:
                print(f"[TRAIN] {name}")
            model.train(**train_kwargs)
            return name
        except RuntimeError as exc:
            last_exc = exc
            if is_oom_error(exc) and idx < len(fallback_batches):
                print(f"[OOM] {name} with batch={try_batch}. Trying smaller batch.")
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Training failed for run: {name}")


def run_one_in_subprocess(
    model_ckpt: str,
    imgsz: int,
    batch: int,
    lr_mult: float,
    project: str,
    epochs: int,
) -> bool:
    name = run_name(model_ckpt, imgsz, batch, lr_mult)
    if (run_dir(project, name) / "results.csv").exists():
        print(f"[SKIP] {name} already exists.")
        return True

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single",
        "--model",
        model_ckpt,
        "--imgsz",
        str(imgsz),
        "--batch",
        str(batch),
        "--lr-mult",
        str(lr_mult),
        "--project",
        project,
        "--epochs",
        str(epochs),
    ]
    print(f"[SPAWN] {name}")
    completed = subprocess.run(cmd)
    if completed.returncode == 0:
        return True

    if completed.returncode < 0:
        print(f"[FAIL] {name} terminated by signal {-completed.returncode} (likely OOM kill).")
    else:
        print(f"[FAIL] {name} exited with code {completed.returncode}.")
    return False
# Candidate columns (Ultralytics results.csv naming varies slightly by version)
COL_VAL_LOSS = [
    "val/box_loss", "metrics/val_box_loss", "val/box_loss"
]
COL_MAP5095 = [
    "metrics/mAP50-95(B)", "metrics/mAP50-95", "metrics/mAP50-95(B)", "metrics/mAP50-95"
]
COL_MAP50 = [
    "metrics/mAP50(B)", "metrics/mAP50", "metrics/mAP50(B)", "metrics/mAP50"
]
COL_PREC = [
    "metrics/precision(B)", "metrics/precision", "metrics/precision(B)"
]
COL_RECALL = [
    "metrics/recall(B)", "metrics/recall", "metrics/recall(B)"
]
COL_TRAIN_BOX = ["train/box_loss"]
COL_TRAIN_CLS = ["train/cls_loss"]
COL_TRAIN_DFL = ["train/dfl_loss"]

def plot_required_for_run(project: str, name: str):
    try:
        df = load_results_csv(project, name)
    except FileNotFoundError as exc:
        print(f"[SKIP PLOT] {exc}")
        return

    # Filter epochs 1..15 (already should be 15, but keeps it explicit)
    df = df[(df["epoch_1based"] >= 1) & (df["epoch_1based"] <= 15)].copy()

    val_box_col = pick_col(df, COL_VAL_LOSS)
    map5095_col = pick_col(df, COL_MAP5095)
    map50_col   = pick_col(df, COL_MAP50)
    prec_col    = pick_col(df, COL_PREC)
    rec_col     = pick_col(df, COL_RECALL)
    train_box   = pick_col(df, COL_TRAIN_BOX)
    train_cls   = pick_col(df, COL_TRAIN_CLS)
    train_dfl   = pick_col(df, COL_TRAIN_DFL)

    print(f"\n=== {name} ===")
    print("Columns found:",
          {"val_box": val_box_col, "mAP50-95": map5095_col, "mAP50": map50_col,
           "prec": prec_col, "recall": rec_col,
           "train_box": train_box, "train_cls": train_cls, "train_dfl": train_dfl})

    # 1) Validation loss vs epochs
    if val_box_col:
        plt.figure()
        plot_metric(df, val_box_col, f"{name}: Validation box_loss vs Epochs", label="val box_loss")
        finalize_plot("Loss")
    else:
        print("Could not find a validation loss column for this run.")

    # 2) mAP50–95 vs epochs
    if map5095_col:
        plt.figure()
        plot_metric(df, map5095_col, f"{name}: mAP50-95 vs Epochs", label="mAP50-95")
        finalize_plot("mAP50-95")
    else:
        print("Could not find mAP50-95 column for this run.")

    # 3) box_loss, cls_loss, dfl_loss, Precision, Recall, mAP50 vs epochs
    plt.figure()
    did_any = False
    for col, lab in [(train_box, "train box_loss"), (train_cls, "train cls_loss"), (train_dfl, "train dfl_loss")]:
        if col:
            plt.plot(df["epoch_1based"], df[col], marker="o", linewidth=1.5, label=lab)
            did_any = True
    if did_any:
        plt.title(f"{name}: Train losses vs Epochs")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()
    else:
        print("Could not find train losses (box/cls/dfl) columns.")

    plt.figure()
    did_any = False
    for col, lab in [(prec_col, "Precision"), (rec_col, "Recall"), (map50_col, "mAP50")]:
        if col:
            plt.plot(df["epoch_1based"], df[col], marker="o", linewidth=1.5, label=lab)
            did_any = True
    if did_any:
        plt.title(f"{name}: Precision / Recall / mAP50 vs Epochs")
        plt.xlabel("Epoch"); plt.ylabel("Metric"); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()
    else:
        print("Could not find Precision/Recall/mAP50 columns.")

def compare_runs(project: str, run_names: list, title: str):
    # Load dfs
    dfs = {}
    for rn in run_names:
        try:
            df = load_results_csv(project, rn)
        except FileNotFoundError:
            print(f"[{title}] Skipping missing run: {rn}")
            continue
        df = df[(df["epoch_1based"] >= 1) & (df["epoch_1based"] <= 15)].copy()
        dfs[rn] = df

    if not dfs:
        print(f"[{title}] No completed runs available to compare.")
        return

    # Determine columns from the first df that has them
    any_df = next(iter(dfs.values()))
    val_box_col = pick_col(any_df, COL_VAL_LOSS) or pick_col(any_df, ["val/box_loss", "metrics/val_box_loss"])
    map5095_col = pick_col(any_df, COL_MAP5095)
    map50_col   = pick_col(any_df, COL_MAP50)

    # Validation loss comparison
    if val_box_col:
        plt.figure()
        for rn, df in dfs.items():
            if val_box_col in df.columns:
                plot_metric(df, val_box_col, f"{title}: Validation Loss", label=rn)
        finalize_plot("Val loss")
    else:
        print(f"[{title}] Could not find a validation loss column.")

    # Accuracy comparison (mAP50-95)
    if map5095_col:
        plt.figure()
        for rn, df in dfs.items():
            if map5095_col in df.columns:
                plot_metric(df, map5095_col, f"{title}: mAP50-95", label=rn)
        finalize_plot("mAP50-95")
    else:
        print(f"[{title}] Could not find mAP50-95 column.")

    # Optional: mAP50 comparison
    if map50_col:
        plt.figure()
        for rn, df in dfs.items():
            if map50_col in df.columns:
                plot_metric(df, map50_col, f"{title}: mAP50", label=rn)
        finalize_plot("mAP50")
def run_sweeps_and_plots():
    all_runs = []
    failed_runs = []

    # --- LR sweep at default batch 16 ---
    for model_ckpt in MODELS:
        for lr_mult in LR_MULTS:
            ok = run_one_in_subprocess(
                model_ckpt=model_ckpt,
                imgsz=IMGSZ,
                batch=16,
                lr_mult=lr_mult,
                project=PROJECT_ROOT,
                epochs=EPOCHS,
            )
            run_id = run_name(model_ckpt, IMGSZ, 16, lr_mult)
            if ok:
                all_runs.append(run_id)
            else:
                failed_runs.append(run_id)

    # --- Batch sweep at default LR (1x) ---
    for model_ckpt in MODELS:
        for b in BATCHES:
            ok = run_one_in_subprocess(
                model_ckpt=model_ckpt,
                imgsz=IMGSZ,
                batch=b,
                lr_mult=1.0,
                project=PROJECT_ROOT,
                epochs=EPOCHS,
            )
            run_id = run_name(model_ckpt, IMGSZ, b, 1.0)
            if ok:
                all_runs.append(run_id)
            else:
                failed_runs.append(run_id)

    print("\nCompleted runs:")
    for r in sorted(set(all_runs)):
        print(" -", r)
    if failed_runs:
        print("\nFailed runs (skipped for plotting/comparison):")
        for r in sorted(set(failed_runs)):
            print(" -", r)

    for name in sorted(set(all_runs)):
        plot_required_for_run(PROJECT_ROOT, name)

    # LR sweep groups: for each model, compare lr1x vs lr5x vs lr0.2x with batch fixed at 16
    for model_ckpt in MODELS:
        base = Path(model_ckpt).stem
        lr_group = [
            run_name(model_ckpt, IMGSZ, 16, 1.0),
            run_name(model_ckpt, IMGSZ, 16, 5.0),
            run_name(model_ckpt, IMGSZ, 16, 0.2),
        ]
        compare_runs(PROJECT_ROOT, lr_group, title=f"{base} LR Sweep (batch=16)")

    # Batch sweep groups: for each model, compare b8 vs b16 vs b32 with lr fixed at 1x
    for model_ckpt in MODELS:
        base = Path(model_ckpt).stem
        batch_group = [
            run_name(model_ckpt, IMGSZ, 8, 1.0),
            run_name(model_ckpt, IMGSZ, 16, 1.0),
            run_name(model_ckpt, IMGSZ, 32, 1.0),
        ]
        compare_runs(PROJECT_ROOT, batch_group, title=f"{base} Batch Sweep (lr=1x)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", action="store_true", help="Run exactly one train job and exit.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr-mult", type=float, default=1.0)
    parser.add_argument("--project", default=PROJECT_ROOT)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.single:
        if not args.model:
            raise ValueError("--model is required when using --single")
        try:
            train_one(
                model_ckpt=args.model,
                imgsz=args.imgsz,
                batch=args.batch,
                lr_mult=args.lr_mult,
                project=args.project,
                epochs=args.epochs,
            )
            return
        except Exception:
            traceback.print_exc()
            raise

    run_sweeps_and_plots()


if __name__ == "__main__":
    main()

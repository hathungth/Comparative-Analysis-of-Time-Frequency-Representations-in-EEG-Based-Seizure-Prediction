import numpy as np
from tqdm import tqdm
from pathlib import Path


def undersample_one_file(X, y, neg_to_pos_ratio=3, max_neg_if_no_pos=0, random_state=42):
    rng = np.random.default_rng(random_state)

    idx_neg = np.where(y == 0)[0]
    idx_pos = np.where(y == 1)[0]

    if len(idx_neg) == 0 and len(idx_pos) == 0:
        return None, None

    keep_idx = list(idx_pos)  # giữ toàn bộ positive

    if len(idx_pos) > 0:
        n_neg_keep = min(len(idx_neg), len(idx_pos) * neg_to_pos_ratio)
    else:
        # file không có positive
        n_neg_keep = min(len(idx_neg), max_neg_if_no_pos)

    if n_neg_keep > 0:
        idx_neg_keep = rng.choice(idx_neg, size=n_neg_keep, replace=False)
        keep_idx.extend(idx_neg_keep.tolist())

    if len(keep_idx) == 0:
        return None, None

    keep_idx = np.array(keep_idx, dtype=np.int64)
    rng.shuffle(keep_idx)

    return X[keep_idx], y[keep_idx]


def enforce_global_ratio(X, y, neg_to_pos_ratio=2, random_state=42):
    """
    Ép tỷ lệ toàn cục class 0 : class 1 đúng bằng neg_to_pos_ratio : 1
    """
    rng = np.random.default_rng(random_state)

    idx_neg = np.where(y == 0)[0]
    idx_pos = np.where(y == 1)[0]

    if len(idx_pos) == 0:
        raise ValueError("No positive samples found after per-file undersampling.")
    if len(idx_neg) == 0:
        raise ValueError("No negative samples found after per-file undersampling.")

    n_neg_keep = min(len(idx_neg), len(idx_pos) * neg_to_pos_ratio)
    idx_neg_keep = rng.choice(idx_neg, size=n_neg_keep, replace=False)

    keep_idx = np.concatenate([idx_pos, idx_neg_keep])
    rng.shuffle(keep_idx)

    return X[keep_idx], y[keep_idx]


def load_and_undersample_by_file(
    npz_root,
    neg_to_pos_ratio=3,
    max_neg_if_no_pos=0,
    random_state=42
):
    X_all, y_all = [], []
    sfreq_ref = None

    npz_files = sorted(Path(npz_root).rglob("*.npz"))
    if len(npz_files) == 0:
        raise ValueError(f"No .npz files found in: {npz_root}")

    total_before_neg = 0
    total_before_pos = 0
    total_after_neg = 0
    total_after_pos = 0

    for i, path in enumerate(tqdm(npz_files, desc="Processing undersample files")):
        data = np.load(path)

        X = data["X"]
        y = data["y"]

        if len(y) == 0:
            continue

        if "sfreq" in data:
            sfreq = float(data["sfreq"])
            if sfreq_ref is None:
                sfreq_ref = sfreq

        n_neg_before = int((y == 0).sum())
        n_pos_before = int((y == 1).sum())

        total_before_neg += n_neg_before
        total_before_pos += n_pos_before

        X_sub, y_sub = undersample_one_file(
            X,
            y,
            neg_to_pos_ratio=neg_to_pos_ratio,
            max_neg_if_no_pos=max_neg_if_no_pos,
            random_state=random_state + i
        )

        if X_sub is None:
            continue

        n_neg_after = int((y_sub == 0).sum())
        n_pos_after = int((y_sub == 1).sum())

        total_after_neg += n_neg_after
        total_after_pos += n_pos_after

        X_all.append(X_sub)
        y_all.append(y_sub)

    if len(X_all) == 0:
        raise ValueError(f"All npz files are empty after undersampling in: {npz_root}")

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    stats = {
        "before_neg": total_before_neg,
        "before_pos": total_before_pos,
        "after_file_neg": total_after_neg,
        "after_file_pos": total_after_pos,
    }

    return X_all, y_all, sfreq_ref, stats


def save_balanced_npz(save_path, X, y, sfreq=None):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if sfreq is None:
        np.savez_compressed(save_path, X=X, y=y)
    else:
        np.savez_compressed(save_path, X=X, y=y, sfreq=sfreq)

    print(f"Saved: {save_path}")


if __name__ == "__main__":
    neg_to_pos_ratio = 1
    input_root  = r"C:\Users\Admin\Documents\Thesis\chbmit-30min\test"
    output_file = rf"C:\Users\Admin\Documents\Thesis\chbmit-store-30m\test_undersampled_ratio_1_{neg_to_pos_ratio}.npz"

    # Bước 1: undersample theo file
    X_train_bal, y_train_bal, sfreq, stats = load_and_undersample_by_file(
        npz_root=input_root,
        neg_to_pos_ratio=neg_to_pos_ratio,
        max_neg_if_no_pos=150,   # file không có positive lấy 100
        random_state=42
    )

    print("Before undersampling:")
    print("  Class 0:", stats["before_neg"])
    print("  Class 1:", stats["before_pos"])

    print("\nAfter per-file undersampling:")
    print("  X shape:", X_train_bal.shape)
    print("  y shape:", y_train_bal.shape)
    print("  Class 0:", stats["after_file_neg"])
    print("  Class 1:", stats["after_file_pos"])

    # Bước 2: ép tỷ lệ toàn cục đúng 3:1
    X_train_bal, y_train_bal = enforce_global_ratio(
        X_train_bal,
        y_train_bal,
        neg_to_pos_ratio=neg_to_pos_ratio,
        random_state=42
    )

    print("\nAfter global ratio enforcement:")
    print("  X shape:", X_train_bal.shape)
    print("  y shape:", y_train_bal.shape)
    print("  Class 0:", int((y_train_bal == 0).sum()))
    print("  Class 1:", int((y_train_bal == 1).sum()))

    save_balanced_npz(output_file, X_train_bal, y_train_bal, sfreq)
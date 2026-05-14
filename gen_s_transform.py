#!/usr/bin/env python3
"""
Sinh S-transform (Stockwell Transform) image từ các file .npz EEG.

Pipeline mỗi window:
  1. Z-score normalize từng channel (trừ mean, chia std)
  2. FFT toàn bộ tín hiệu: X = FFT(x), shape (T,)
  3. Với mỗi tần số f_k trong [F_MIN, F_MAX]:
       a. Dịch phổ: shifted_X = roll(X, j_k) với j_k = round(f_k * T / fs)
       b. Nhân với Gaussian trong miền tần số: G[n] = exp(-2π²n²/j_k²)
       c. IFFT(shifted_X * G) → hàng S-transform tại f_k, shape (T,)
  4. Power spectrum: |S|², shape (n_freqs, T)
  5. Log transform: log1p(power)
  6. Ghép dọc 18 channels → ảnh (18*n_freqs, T)
  7. Min-max normalize → uint8 [0, 255]
  8. Resize về 224×224 (BILINEAR) và lưu JPEG

TỐI ƯU:
  - Gaussian filterbank tính 1 lần duy nhất per NPZ (không tính lại cho mỗi window/channel)
  - ProcessPoolExecutor: xử lý song song nhiều chunk windows trên nhiều CPU
  - OVERWRITE_IMAGES = False: bỏ qua ảnh đã tồn tại khi chạy lại
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


# =========================================================
# CẤU HÌNH
# =========================================================
neg_to_pos_ratio = 1 
NPZ_DIR = Path(rf"C:\Users\Admin\Documents\Thesis\chbmit-store-30m")

NPZ_PATTERN = "*.npz"

# Số CPU dùng để song song hóa (None = tự động = os.cpu_count())
NUM_WORKERS = None
CHUNK_SIZE  = 64     # số window mỗi chunk gửi cho 1 worker

# Chế độ lưu
SAVE_AS_IMAGES   = True
SAVE_AS_NPZ      = False
OVERWRITE_IMAGES = False   # False = bỏ qua ảnh đã có → nhanh hơn khi chạy lại

IMAGE_EXT    = ".jpg"
JPEG_QUALITY = 95

# =========================================================
# S-TRANSFORM CONFIG
# =========================================================

FS             = None         # None → đọc từ sfreq trong npz
TILE_SIZE      = 224
USE_LOG_POWER  = True

F_MIN          = 13       # tần số thấp nhất phân tích (Hz)
F_MAX          = 40       # tần số cao nhất phân tích (Hz)
N_FREQS        = 32       # số tần số phân tích trong [F_MIN, F_MAX]

RESAMPLE_MODE  = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR


# =========================================================
# PRE-COMPUTE S-TRANSFORM PARAMS (1 lần per npz)
# =========================================================

def precompute_s_transform_params(
    actual_fs: float,
    seq_len: int,
    f_min: float,
    f_max: float,
    n_freqs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tính trước Gaussian filterbank và chỉ số dịch FFT cho S-transform, gọi 1 lần per NPZ.

    Nguyên lý S-transform tại tần số f:
      S(τ, f) = IFFT[ roll(FFT(x), j) * G ]
      - j = round(f * T / fs): số bin cần dịch phổ FFT(x) để căn giữa tại f
      - G[n] = exp(-2π²n²/j²): Gaussian trong miền tần số
        với n = chỉ số FFT nguyên [0, 1, ..., T//2, -T//2+1, ..., -1]
      - Độ rộng Gaussian tỉ lệ nghịch với f: tần số cao → cửa sổ thời gian hẹp hơn

    Quy trình:
      - Tạo n_freqs tần số cách đều trong [f_min, f_max]
      - Với mỗi f_k: tính j_k = round(f_k * seq_len / fs)
      - Tính n_arr = chỉ số FFT nguyên của tín hiệu độ dài seq_len
      - Với mỗi j_k ≠ 0: G[k, n] = exp(-2π²n²/j_k²)
      - Với j_k = 0 (DC): G[k, :] = 1.0 (cửa sổ hình chữ nhật, không lọc)

    Tham số:
      actual_fs : tần số lấy mẫu (Hz)
      seq_len   : số mẫu mỗi window (= T, chiều thời gian của X)
      f_min     : tần số thấp nhất cần phân tích (Hz)
      f_max     : tần số cao nhất cần phân tích (Hz)
      n_freqs   : số tần số phân tích (= số hàng của ma trận S-transform)

    Trả về:
      gaussians  : ndarray shape (n_freqs, seq_len), dtype float32
                   Gaussian trong miền tần số cho từng tần số mục tiêu.
      shift_idxs : ndarray shape (n_freqs,), dtype int32
                   Số bin cần roll(FFT(x), j) để dịch phổ đến tần số f_k.
    """
    nyquist = actual_fs / 2.0
    f_min   = max(1e-6, float(f_min))     # tránh j=0 gây cửa sổ phẳng
    f_max   = min(float(f_max), nyquist)
    if f_min >= f_max:
        raise ValueError(f"Dải tần không hợp lệ: f_min={f_min} >= f_max={f_max} (Nyquist={nyquist:.1f})")

    # Chỉ số FFT nguyên: [0, 1, ..., T//2, -T//2+1, ..., -1]
    n_arr = np.round(np.fft.fftfreq(seq_len) * seq_len).astype(np.float64)

    target_freqs = np.linspace(f_min, f_max, n_freqs)
    shift_idxs   = np.round(target_freqs * seq_len / actual_fs).astype(np.int32)

    gaussians = np.empty((n_freqs, seq_len), dtype=np.float32)
    for k, j in enumerate(shift_idxs):
        if j == 0:
            gaussians[k, :] = 1.0
        else:
            gaussians[k, :] = np.exp(-2.0 * np.pi**2 * n_arr**2 / j**2).astype(np.float32)

    return gaussians, shift_idxs


# =========================================================
# WORKER (chạy trong process riêng)
# =========================================================

def _worker_chunk(args: tuple) -> List[Tuple[int, np.ndarray]]:
    """
    Xử lý 1 chunk windows liên tiếp trong process con.
    Index ảnh đầu ra tính từ start_idx (dùng khi idx_slice liên tục).
    Trả về: list of (window_index, gray_img uint8 HxW)
    """
    (
        start_idx,
        X_chunk,     # (chunk_size, C, T) float32 — dữ liệu EEG đã cắt window
        gaussians,   # (n_freqs, T) float32 — Gaussian filter tính sẵn cho từng tần số
        shift_idxs,  # (n_freqs,) int32 — số bin roll(FFT(x), j) cho từng tần số
        use_log_power,
        tile_size,
    ) = args

    results = []

    for local_i, window_data in enumerate(X_chunk):
        # Z-score normalize từng channel: (window_data - mean) / std
        mean = window_data.mean(axis=1, keepdims=True)
        std  = window_data.std(axis=1, keepdims=True)
        window_norm = (window_data - mean) / (std + 1e-8)   # (C, T)

        channel_maps = []
        for ch_signal in window_norm:
            # FFT toàn bộ tín hiệu 1 lần: X shape (T,), phức
            X = np.fft.fft(ch_signal)

            # Dịch phổ đến từng tần số mục tiêu, nhân Gaussian, IFFT vectorized
            # shifted_X: (n_freqs, T) — roll(X, j_k) cho mỗi j_k trong shift_idxs
            shifted_X = np.array([np.roll(X, j) for j in shift_idxs])
            # S-transform: IFFT theo trục thời gian → (n_freqs, T), phức
            S = np.fft.ifft(shifted_X * gaussians, axis=1)
            power = np.abs(S) ** 2   # (n_freqs, T), thực
            if use_log_power:
                power = np.log1p(power)
            channel_maps.append(power[::-1, :].astype(np.float32))   # flip: tần số cao → thấp

        # Ghép dọc 18 channels → (C*n_freqs, T)
        vertical = np.concatenate(channel_maps, axis=0)

        # Min-max normalize → uint8 [0, 255]
        vmin, vmax = vertical.min(), vertical.max()
        if vmax - vmin < 1e-8:
            img_u8 = np.zeros((tile_size, tile_size), dtype=np.uint8)
        else:
            img_u8 = ((vertical - vmin) / (vmax - vmin) * 255).clip(0, 255).astype(np.uint8)
            img_u8 = np.array(
                Image.fromarray(img_u8, mode="L").resize((tile_size, tile_size), RESAMPLE_MODE),
                dtype=np.uint8,
            )

        results.append((start_idx + local_i, img_u8))

    return results


# =========================================================
# PROCESS ONE NPZ
# =========================================================

def process_one_npz(npz_path: Path, num_workers: int) -> dict:
    print(f"\n[INFO] {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        if "X" not in data:
            raise KeyError(f"Không có key 'X' trong {npz_path.name}")
        X   = data["X"].astype(np.float32)   # (N, C, T)
        y   = data["y"].astype(np.int64) if "y" in data else None
        meta_keys = ["sfreq", "window_start_sec", "window_end_sec", "source_edf", "patient_id"]
        meta = {k: data[k] for k in meta_keys if k in data}

    if X.ndim != 3:
        raise ValueError(f"X phải có shape (N, C, T), nhận được {X.shape}")

    N, n_channels, seq_len = X.shape
    actual_fs = float(meta["sfreq"]) if FS is None else float(FS)
    print(f"    shape={X.shape}  fs={actual_fs}Hz  workers={num_workers}")

    rel = npz_path.relative_to(NPZ_DIR)
    wavelet_dir = NPZ_DIR / rel.with_suffix("")
    wavelet_dir.mkdir(parents=True, exist_ok=True)

    # Tính Gaussian filterbank và shift_idxs 1 lần cho toàn bộ NPZ — dùng chung cho mọi window và channel
    gaussians, shift_idxs = precompute_s_transform_params(actual_fs, seq_len, F_MIN, F_MAX, N_FREQS)

    # Kiểm tra những ảnh nào đã tồn tại (để skip)
    if not OVERWRITE_IMAGES:
        existing = {p.stem for p in wavelet_dir.glob(f"image_*{IMAGE_EXT}")}
        todo_indices = [i for i in range(N) if f"image_{i:05d}" not in existing]
        if len(todo_indices) < N:
            print(f"    Bỏ qua {N - len(todo_indices)} ảnh đã có, gen thêm {len(todo_indices)}")
    else:
        todo_indices = list(range(N))

    if not todo_indices:
        print(f"    Tất cả {N} ảnh đã tồn tại, skip.")
        return {"npz_path": str(npz_path), "wavelet_dir": str(wavelet_dir), "num_samples": N}

    # Chia todo_indices thành chunks
    chunks = []
    for chunk_start in range(0, len(todo_indices), CHUNK_SIZE):
        idx_slice = todo_indices[chunk_start : chunk_start + CHUNK_SIZE]
        X_chunk   = X[idx_slice]     # copy slice (sẽ serialize sang worker)
        chunks.append((
            idx_slice, X_chunk,
            gaussians, shift_idxs,
            USE_LOG_POWER, TILE_SIZE,
        ))

    def _submit(executor):
        return {executor.submit(_worker_chunk_with_idxs, c): c for c in chunks}

    saved = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_worker_chunk_with_idxs, c): c for c in chunks}
        with tqdm(total=len(todo_indices), desc=npz_path.stem, leave=False) as pbar:
            for future in as_completed(futures):
                for win_idx, img_u8 in future.result():
                    save_path = wavelet_dir / f"image_{win_idx:05d}{IMAGE_EXT}"
                    _save_image(img_u8, save_path)
                    saved += 1
                    pbar.update(1)

    print(f"    Đã lưu {saved} ảnh → {wavelet_dir}")

    # Lưu NPZ nếu cần
    if SAVE_AS_NPZ:
        _save_wavelet_npz(wavelet_dir, npz_path, N, y, meta)

    return {"npz_path": str(npz_path), "wavelet_dir": str(wavelet_dir), "num_samples": N}


def _worker_chunk_with_idxs(args: tuple) -> List[Tuple[int, np.ndarray]]:
    """
    Xử lý 1 chunk windows trong process con.
    Nhận idx_slice thực — hỗ trợ index không liên tục (ảnh đã skip một phần).
    Trả về: list of (window_index, gray_img uint8 HxW)
    """
    (
        idx_slice,
        X_chunk,     # (chunk_size, C, T) float32 — dữ liệu EEG đã cắt window
        gaussians,   # (n_freqs, T) float32 — Gaussian filter tính sẵn cho từng tần số
        shift_idxs,  # (n_freqs,) int32 — số bin roll(FFT(x), j) cho từng tần số
        use_log_power,
        tile_size,
    ) = args

    results = []

    for win_idx, window_data in zip(idx_slice, X_chunk):
        # Z-score normalize từng channel: (window_data - mean) / std
        mean = window_data.mean(axis=1, keepdims=True)
        std  = window_data.std(axis=1, keepdims=True)
        window_norm = (window_data - mean) / (std + 1e-8)

        channel_maps = []
        for ch_signal in window_norm:
            # FFT toàn bộ tín hiệu 1 lần: X shape (T,), phức
            X = np.fft.fft(ch_signal)

            # Dịch phổ đến từng tần số mục tiêu, nhân Gaussian, IFFT vectorized
            # shifted_X: (n_freqs, T) — roll(X, j_k) cho mỗi j_k trong shift_idxs
            shifted_X = np.array([np.roll(X, j) for j in shift_idxs])
            # S-transform: IFFT theo trục thời gian → (n_freqs, T), phức
            S = np.fft.ifft(shifted_X * gaussians, axis=1)
            power = np.abs(S) ** 2   # (n_freqs, T), thực
            if use_log_power:
                power = np.log1p(power)
            channel_maps.append(power[::-1, :].astype(np.float32))   # flip: tần số cao → thấp

        # Ghép dọc 18 channels → (C*n_freqs, T)
        vertical = np.concatenate(channel_maps, axis=0)
        vmin, vmax = vertical.min(), vertical.max()

        # Min-max normalize → uint8 [0, 255]
        if vmax - vmin < 1e-8:
            img_u8 = np.zeros((tile_size, tile_size), dtype=np.uint8)
        else:
            img_u8 = ((vertical - vmin) / (vmax - vmin) * 255).clip(0, 255).astype(np.uint8)
            img_u8 = np.array(
                Image.fromarray(img_u8, mode="L").resize(
                    (tile_size, tile_size),
                    Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR,
                ),
                dtype=np.uint8,
            )

        results.append((win_idx, img_u8))

    return results


def _save_image(img_u8: np.ndarray, save_path: Path) -> None:
    ext = save_path.suffix.lower()
    pil_img = Image.fromarray(img_u8, mode="L")
    if ext in (".jpg", ".jpeg"):
        pil_img.save(save_path, format="JPEG", quality=JPEG_QUALITY)
    elif ext == ".png":
        pil_img.save(save_path, format="PNG")
    else:
        pil_img.save(save_path)


def _save_wavelet_npz(
    wavelet_dir: Path, npz_path: Path, N: int,
    y: Optional[np.ndarray], meta: dict,
) -> None:
    """Gom tất cả ảnh đã lưu vào 1 file NPZ."""
    imgs = []
    for i in range(N):
        p = wavelet_dir / f"image_{i:05d}{IMAGE_EXT}"
        if p.exists():
            imgs.append(np.array(Image.open(p), dtype=np.uint8)[None])   # (1, H, W)
    if not imgs:
        return
    X_wav = np.stack(imgs, axis=0).astype(np.uint8)   # (N, 1, H, W)
    save_dict = {"X": X_wav}
    if y is not None:
        save_dict["y"] = y
    save_dict.update(meta)
    out = wavelet_dir / f"{npz_path.stem}_wavelet.npz"
    np.savez_compressed(out, **save_dict)
    print(f"    Saved NPZ: {out}  shape={X_wav.shape}")


# =========================================================
# PROCESS ROOT
# =========================================================

def process_root(num_workers: int) -> tuple:
    npz_files = sorted([
        p for p in NPZ_DIR.rglob(NPZ_PATTERN)
        if p.is_file() and not p.name.endswith("_wavelet.npz")
    ])

    if not npz_files:
        print(f"  [WARNING] Không có .npz nào trong: {NPZ_DIR}")
        return [], []

    print(f"  Tìm thấy {len(npz_files)} file .npz")

    results, failed = [], []
    for npz_path in npz_files:
        try:
            results.append(process_one_npz(npz_path, num_workers))
        except Exception as e:
            failed.append((str(npz_path), str(e)))
            print(f"  [ERROR] {npz_path.name}: {e}")

    return results, failed


# =========================================================
# MAIN
# =========================================================

def main():
    if not NPZ_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy NPZ_DIR: {NPZ_DIR}")

    num_workers = NUM_WORKERS or os.cpu_count() or 4
    print(f"NPZ_DIR      : {NPZ_DIR}")
    print(f"Output dir   : {NPZ_DIR}")
    print(f"NUM_WORKERS  : {num_workers}")
    print(f"CHUNK_SIZE   : {CHUNK_SIZE}")
    print(f"OVERWRITE    : {OVERWRITE_IMAGES}")
    print(f"S-transform  : {F_MIN}–{F_MAX} Hz  n_freqs={N_FREQS}  tile={TILE_SIZE}px")

    print(f"\n{'='*60}")
    print(f"{'='*60}")

    results, failed = process_root(num_workers)

    total_ok      = len(results)
    total_fail    = len(failed)
    total_windows = sum(r["num_samples"] for r in results)

    if failed:
        print(f"\n  Failed ({len(failed)}):")
        for path, err in failed:
            print(f"    - {Path(path).name}: {err}")

    print(f"\n{'='*60}")
    print(f"  TỔNG KẾT")
    print(f"  Files OK   : {total_ok}")
    print(f"  Files FAIL : {total_fail}")
    print(f"  Tổng ảnh   : {total_windows}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

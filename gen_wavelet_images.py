#!/usr/bin/env python3
"""
Sinh wavelet scalogram image từ các file .npz EEG.

TỐI ƯU:
  - Scales CWT tính 1 lần duy nhất per NPZ (không tính lại cho mỗi window/channel)
  - ProcessPoolExecutor: xử lý song song nhiều chunk windows trên nhiều CPU
  - OVERWRITE_IMAGES = False: bỏ qua ảnh đã tồn tại khi chạy lại
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pywt
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
# WAVELET CONFIG
# =========================================================

FS             = None         # None → đọc từ sfreq trong npz
WAVELET_NAME   = "cmor1.5-1.0"
TILE_SIZE      = 224
USE_LOG_POWER  = True

BAND_LOW       = 13
BAND_HIGH      = 40
N_FREQS        = 32

RESAMPLE_MODE  = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR


# =========================================================
# PRE-COMPUTE SCALES (1 lần per npz)
# =========================================================

def precompute_scales(
    actual_fs: float,
    band_low: float,
    band_high: float,
    n_freqs: int,
    wavelet_name: str,
) -> np.ndarray:
    """
    Tính CWT scales 1 lần.
    Scales chỉ phụ thuộc vào fs, band, n_freqs — không đổi qua các windows.
    """
    nyquist   = actual_fs / 2.0 - 1e-6
    band_low  = max(1e-6, float(band_low))
    band_high = min(float(band_high), nyquist)

    if band_low >= band_high:
        raise ValueError(f"Band không hợp lệ: {band_low} >= {band_high} (Nyquist={nyquist:.1f})")

    freqs_hz = np.linspace(band_low, band_high, n_freqs, dtype=np.float32)
    scales   = pywt.frequency2scale(wavelet_name, freqs_hz / actual_fs).astype(np.float32)
    return scales


# =========================================================
# WORKER (chạy trong process riêng)
# =========================================================

def _worker_chunk(args: tuple) -> List[Tuple[int, np.ndarray]]:
    """
    Xử lý 1 chunk windows trong process con.
    Trả về: list of (window_index, gray_img uint8 HxW)
    """
    (
        start_idx,
        X_chunk,      # (chunk_size, C, T) float32
        scales,       # (n_freqs,) float32
        wavelet_name,
        actual_fs,
        use_log_power,
        tile_size,
    ) = args

    results = []

    for local_i, window_data in enumerate(X_chunk):
        # z-score normalize từng channel
        mean = window_data.mean(axis=1, keepdims=True)
        std  = window_data.std(axis=1, keepdims=True)
        window_norm = (window_data - mean) / (std + 1e-8)   # (C, T)

        # CWT cho tất cả channels — scales đã tính sẵn, không tính lại
        channel_maps = []
        for ch_signal in window_norm:
            coeffs, _ = pywt.cwt(
                ch_signal, scales, wavelet_name,
                sampling_period=1.0 / actual_fs, method="fft",
            )
            power = np.abs(coeffs) ** 2
            if use_log_power:
                power = np.log1p(power)
            channel_maps.append(power[::-1, :].astype(np.float32))   # flip freq

        # Ghép dọc các channel
        vertical = np.concatenate(channel_maps, axis=0)   # (C*n_freqs, T)

        # Normalize → uint8
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

    # Tính scales 1 lần cho toàn bộ NPZ
    scales = precompute_scales(actual_fs, BAND_LOW, BAND_HIGH, N_FREQS, WAVELET_NAME)

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
            idx_slice[0],            # start_idx để đặt tên file
            X_chunk,
            scales,
            WAVELET_NAME,
            actual_fs,
            USE_LOG_POWER,
            TILE_SIZE,
        ))
        # Đính kèm idx_slice thực tế (có thể không liên tục)
        chunks[-1] = (idx_slice, X_chunk, scales, WAVELET_NAME, actual_fs, USE_LOG_POWER, TILE_SIZE)

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
    Giống _worker_chunk nhưng nhận idx_slice thực (không liên tục).
    """
    (
        idx_slice,    # list of actual window indices
        X_chunk,
        scales,
        wavelet_name,
        actual_fs,
        use_log_power,
        tile_size,
    ) = args

    results = []

    for win_idx, window_data in zip(idx_slice, X_chunk):
        mean = window_data.mean(axis=1, keepdims=True)
        std  = window_data.std(axis=1, keepdims=True)
        window_norm = (window_data - mean) / (std + 1e-8)

        channel_maps = []
        for ch_signal in window_norm:
            coeffs, _ = pywt.cwt(
                ch_signal, scales, wavelet_name,
                sampling_period=1.0 / actual_fs, method="fft",
            )
            power = np.abs(coeffs) ** 2
            if use_log_power:
                power = np.log1p(power)
            channel_maps.append(power[::-1, :].astype(np.float32))

        vertical = np.concatenate(channel_maps, axis=0)
        vmin, vmax = vertical.min(), vertical.max()

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
    print(f"Band         : {BAND_LOW}–{BAND_HIGH} Hz  n_freqs={N_FREQS}")
    print(f"Wavelet      : {WAVELET_NAME}  tile={TILE_SIZE}px")

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

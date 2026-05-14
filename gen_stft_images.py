#!/usr/bin/env python3
"""
Sinh stft spectrogram image từ các file .npz EEG.
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import stft as scipy_stft
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
N_FFT          = 256      # FFT points → độ phân giải tần số = fs/N_FFT ≈ 1 Hz tại 256 Hz
HOP_LENGTH     = 32       # bước nhảy giữa các frame = N_FFT//8 → overlap ~87.5%
WIN_LENGTH     = 256      # độ dài cửa sổ phân tích (bằng N_FFT)
WINDOW_TYPE    = "hann"   # cửa sổ Hann tiêu chuẩn
TILE_SIZE      = 224
USE_LOG_POWER  = True

BAND_LOW       = 13
BAND_HIGH      = 40
N_FREQS        = 32

RESAMPLE_MODE  = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR


# =========================================================
# PRE-COMPUTE STFT PARAMS (1 lần per npz)
# =========================================================

def precompute_stft_params(
    actual_fs: float,
    band_low: float,
    band_high: float,
    n_fft: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Tính trước mask tần số STFT, gọi 1 lần per NPZ."""
    nyquist   = actual_fs / 2.0
    band_low  = max(0.0, float(band_low))
    band_high = min(float(band_high), nyquist)
    if band_low >= band_high:
        raise ValueError(f"Band không hợp lệ: {band_low} >= {band_high} (Nyquist={nyquist:.1f})")
    freqs_all = np.fft.rfftfreq(n_fft, d=1.0 / actual_fs).astype(np.float32)
    band_mask = (freqs_all >= band_low) & (freqs_all <= band_high)
    if band_mask.sum() < 2:
        raise ValueError(f"Quá ít bin tần số trong [{band_low}, {band_high}] Hz. Tăng N_FFT.")
    return freqs_all, band_mask


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
        band_mask,
        n_fft,
        hop_length,
        win_length,
        window_type,
        actual_fs,
        n_freqs,
        use_log_power,
        tile_size,
    ) = args

    results = []

    for local_i, window_data in enumerate(X_chunk):
        # z-score normalize từng channel
        mean = window_data.mean(axis=1, keepdims=True)
        std  = window_data.std(axis=1, keepdims=True)
        window_norm = (window_data - mean) / (std + 1e-8)   # (C, T)

        channel_maps = []
        for ch_signal in window_norm:
            _, _, Zxx = scipy_stft(
                ch_signal,
                fs=actual_fs,
                window=window_type,
                nperseg=win_length,
                noverlap=win_length - hop_length,
                nfft=n_fft,
                boundary=None,
                padded=False,
            )
            Zxx_band   = Zxx[band_mask, :]
            power_band = np.abs(Zxx_band) ** 2
            if use_log_power:
                power_band = np.log1p(power_band)
            n_band_bins = power_band.shape[0]
            if n_band_bins != n_freqs:
                src_idx = np.linspace(0, n_band_bins - 1, n_band_bins, dtype=np.float32)
                dst_idx = np.linspace(0, n_band_bins - 1, n_freqs,     dtype=np.float32)
                power_resampled = np.stack(
                    [np.interp(dst_idx, src_idx, power_band[:, t])
                     for t in range(power_band.shape[1])],
                    axis=1,
                ).astype(np.float32)
            else:
                power_resampled = power_band.astype(np.float32)
            channel_maps.append(power_resampled[::-1, :])   # flip freq

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

    # Tính STFT params 1 lần cho toàn bộ NPZ
    freqs_all, band_mask = precompute_stft_params(actual_fs, BAND_LOW, BAND_HIGH, N_FFT)

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
            band_mask, N_FFT, HOP_LENGTH, WIN_LENGTH, WINDOW_TYPE,
            actual_fs, N_FREQS, USE_LOG_POWER, TILE_SIZE,
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
    Giống _worker_chunk nhưng nhận idx_slice thực (không liên tục).
    """
    (
        idx_slice,
        X_chunk,
        band_mask,
        n_fft,
        hop_length,
        win_length,
        window_type,
        actual_fs,
        n_freqs,
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
            _, _, Zxx = scipy_stft(
                ch_signal,
                fs=actual_fs,
                window=window_type,
                nperseg=win_length,
                noverlap=win_length - hop_length,
                nfft=n_fft,
                boundary=None,
                padded=False,
            )
            Zxx_band   = Zxx[band_mask, :]
            power_band = np.abs(Zxx_band) ** 2
            if use_log_power:
                power_band = np.log1p(power_band)
            n_band_bins = power_band.shape[0]
            if n_band_bins != n_freqs:
                src_idx = np.linspace(0, n_band_bins - 1, n_band_bins, dtype=np.float32)
                dst_idx = np.linspace(0, n_band_bins - 1, n_freqs,     dtype=np.float32)
                power_resampled = np.stack(
                    [np.interp(dst_idx, src_idx, power_band[:, t])
                     for t in range(power_band.shape[1])],
                    axis=1,
                ).astype(np.float32)
            else:
                power_resampled = power_band.astype(np.float32)
            channel_maps.append(power_resampled[::-1, :])   # flip freq

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
    print(f"STFT         : N_FFT={N_FFT}  hop={HOP_LENGTH}  win={WIN_LENGTH}  window={WINDOW_TYPE}  tile={TILE_SIZE}px")

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

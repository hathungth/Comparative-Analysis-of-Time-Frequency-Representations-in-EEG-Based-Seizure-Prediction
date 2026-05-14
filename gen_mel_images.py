#!/usr/bin/env python3
"""
Sinh Mel Spectrogram image từ các file .npz EEG.

Pipeline mỗi window:
  1. Z-score normalize từng channel (trừ mean, chia std)
  2. STFT (scipy.signal.stft) → ma trận phức Zxx (n_fft//2+1, n_time_frames)
  3. Tính power spectrum: |Zxx|²
  4. Nhân với mel filterbank (n_mels × n_fft//2+1) → mel power (n_mels, n_time_frames)
  5. Log transform: log1p(mel_power)
  6. Ghép dọc 18 channels → ảnh (18*n_mels, n_time_frames)
  7. Min-max normalize → uint8 [0, 255]
  8. Resize về 224×224 (BILINEAR) và lưu JPEG

TỐI ƯU:
  - Mel filterbank tính 1 lần duy nhất per NPZ (không tính lại cho mỗi window/channel)
  - ProcessPoolExecutor: xử lý song song nhiều chunk windows trên nhiều CPU
  - OVERWRITE_IMAGES = False: bỏ qua ảnh đã tồn tại khi chạy lại
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
# MEL SPECTROGRAM CONFIG
# =========================================================

FS             = None         # None → đọc từ sfreq trong npz
N_FFT          = 256      # FFT points → độ phân giải tần số = fs/N_FFT ≈ 1 Hz tại 256 Hz
HOP_LENGTH     = 32       # bước nhảy giữa các frame = N_FFT//8 → overlap ~87.5%
WIN_LENGTH     = 256      # độ dài cửa sổ phân tích (bằng N_FFT)
WINDOW_TYPE    = "hann"   # cửa sổ Hann tiêu chuẩn
TILE_SIZE      = 224
USE_LOG_POWER  = True

F_MIN          = 13       # tần số thấp nhất của mel filterbank (Hz)
F_MAX          = 40       # tần số cao nhất của mel filterbank (Hz)
N_MELS         = 32       # số mel filter bands

RESAMPLE_MODE  = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR


# =========================================================
# PRE-COMPUTE MEL FILTERBANK (1 lần per npz)
# =========================================================

def precompute_mel_filterbank(
    actual_fs: float,
    n_fft: int,
    n_mels: int,
    f_min: float,
    f_max: float,
) -> np.ndarray:
    """
    Xây dựng ma trận mel filterbank triangular, gọi 1 lần per NPZ.

    Quy trình:
      - Tạo n_mels + 2 điểm cách đều nhau trên thang mel trong khoảng [f_min, f_max]
      - Chuyển ngược về Hz để xác định tần số tâm và biên của từng filter
      - Ánh xạ tần số → chỉ số bin STFT (floor((n_fft+1)*f/fs))
      - Mỗi filter m là tam giác: tăng tuyến tính từ bin[m-1]→bin[m],
        giảm tuyến tính từ bin[m]→bin[m+1], bằng 0 ở ngoài
      - Đỉnh tam giác = 1.0 tại bin trung tâm bin[m]

    Tham số:
      actual_fs : tần số lấy mẫu (Hz)
      n_fft     : số điểm FFT (xác định số bin STFT = n_fft//2 + 1)
      n_mels    : số mel filter bands (= số hàng của filterbank)
      f_min     : tần số thấp nhất của filterbank (Hz)
      f_max     : tần số cao nhất của filterbank (Hz)

    Trả về:
      mel_fb : ndarray shape (n_mels, n_fft//2 + 1), dtype float32
               Nhân mel_fb @ power_stft để thu được mel power.
    """
    nyquist = actual_fs / 2.0
    f_min   = max(0.0, float(f_min))
    f_max   = min(float(f_max), nyquist)
    if f_min >= f_max:
        raise ValueError(f"Dải tần không hợp lệ: f_min={f_min} >= f_max={f_max} (Nyquist={nyquist:.1f})")

    # Chuyển đổi Hz ↔ Mel theo công thức HTK: mel = 2595 * log10(1 + f/700)
    def _hz_to_mel(f: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def _mel_to_hz(m: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    # n_mels + 2 điểm cách đều trên thang mel → chuyển về Hz → chỉ số bin STFT
    mel_points  = np.linspace(_hz_to_mel(f_min), _hz_to_mel(f_max), n_mels + 2)
    hz_points   = _mel_to_hz(mel_points)
    bin_points  = np.floor((n_fft + 1) * hz_points / actual_fs).astype(np.int32)

    n_bins = n_fft // 2 + 1
    bin_points = np.clip(bin_points, 0, n_bins - 1)

    mel_fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for m in range(n_mels):
        lo, mid, hi = bin_points[m], bin_points[m + 1], bin_points[m + 2]
        # Sườn tăng: bin[lo] → bin[mid]
        if mid > lo:
            mel_fb[m, lo:mid] = (np.arange(lo, mid) - lo) / (mid - lo)
        # Đỉnh
        if mid < n_bins:
            mel_fb[m, mid] = 1.0
        # Sườn giảm: bin[mid] → bin[hi]
        if hi > mid:
            mel_fb[m, mid + 1:hi] = (hi - np.arange(mid + 1, hi)) / (hi - mid)

    return mel_fb


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
        X_chunk,   # (chunk_size, C, T) float32 — dữ liệu EEG đã cắt window
        mel_fb,    # (n_mels, n_fft//2+1) float32 — mel filterbank tính sẵn
        n_fft,
        hop_length,
        win_length,
        window_type,
        actual_fs,
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
            # STFT → Zxx shape (n_fft//2+1, n_time_frames), phức
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
            # Power spectrum rồi áp mel filterbank
            # mel_fb @ |Zxx|²: (n_mels, n_fft//2+1) x (n_fft//2+1, n_time_frames)
            #                 → (n_mels, n_time_frames)
            mel_power = mel_fb @ (np.abs(Zxx) ** 2)
            if use_log_power:
                mel_power = np.log1p(mel_power)
            channel_maps.append(mel_power[::-1, :].astype(np.float32))   # flip: tần số cao → thấp

        # Ghép dọc 18 channels → (C*n_mels, n_time_frames)
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

    # Tính mel filterbank 1 lần cho toàn bộ NPZ — dùng chung cho mọi window và channel
    mel_fb = precompute_mel_filterbank(actual_fs, N_FFT, N_MELS, F_MIN, F_MAX)

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
            mel_fb, N_FFT, HOP_LENGTH, WIN_LENGTH, WINDOW_TYPE,
            actual_fs, USE_LOG_POWER, TILE_SIZE,
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
        X_chunk,   # (chunk_size, C, T) float32 — dữ liệu EEG đã cắt window
        mel_fb,    # (n_mels, n_fft//2+1) float32 — mel filterbank tính sẵn
        n_fft,
        hop_length,
        win_length,
        window_type,
        actual_fs,
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
            # STFT → Zxx shape (n_fft//2+1, n_time_frames), phức
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
            # Power spectrum rồi áp mel filterbank
            # mel_fb @ |Zxx|²: (n_mels, n_fft//2+1) x (n_fft//2+1, n_time_frames)
            #                 → (n_mels, n_time_frames)
            mel_power = mel_fb @ (np.abs(Zxx) ** 2)
            if use_log_power:
                mel_power = np.log1p(mel_power)
            channel_maps.append(mel_power[::-1, :].astype(np.float32))   # flip: tần số cao → thấp

        # Ghép dọc 18 channels → (C*n_mels, n_time_frames)
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
    print(f"Mel band     : {F_MIN}–{F_MAX} Hz  n_mels={N_MELS}")
    print(f"STFT config  : N_FFT={N_FFT}  hop={HOP_LENGTH}  win={WIN_LENGTH}  window={WINDOW_TYPE}  tile={TILE_SIZE}px")

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

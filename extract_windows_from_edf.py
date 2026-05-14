import os
import re
import glob
import numpy as np
import mne
import wfdb
from tqdm import tqdm
import warnings
from typing import List, Tuple, Optional

# =========================================================
# 1) CHANNEL CONFIG
# =========================================================
CH_LABELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ'
]


def normalize_channel_name(ch: str) -> str:
    """
    Chuẩn hóa tên kênh để match ổn định hơn:
    - bỏ khoảng trắng
    - uppercase
    - nếu hậu tố cuối là số do hệ thống tự thêm thì bỏ
    """
    ch = ch.strip().upper().replace(" ", "")
    parts = ch.split("-")
    if len(parts) >= 2 and parts[-1].isdigit():
        ch = "-".join(parts[:-1])
    return ch


def pick_required_channels(raw, wanted_channels):
    """
    Chọn đúng các kênh mong muốn và giữ đúng thứ tự.
    Tự loại channel rác trước khi pick.
    """
    junk_channels = [ch for ch in raw.ch_names if ch.strip().startswith("--")]
    if junk_channels:
        raw.drop_channels(junk_channels)

    wanted_norm = [normalize_channel_name(ch) for ch in wanted_channels]

    raw_map = {}
    for ch in raw.ch_names:
        raw_map.setdefault(normalize_channel_name(ch), []).append(ch)

    picks = []
    missing = []

    for ch in wanted_norm:
        if ch in raw_map:
            picks.append(raw_map[ch][0])
        else:
            missing.append(ch)

    if missing:
        raise ValueError(f"Missing channels: {missing}")

    raw.pick(picks)
    return raw


# =========================================================
# 2) READ SEIZURE ANNOTATION
# =========================================================
def read_seizures_sec(edf_path: str) -> List[Tuple[float, float]]:
    """
    Đọc seizure intervals từ file annotation wfdb đi kèm.
    Trả về list[(start_sec, end_sec)].
    Nếu không có annotation hoặc lỗi đọc -> trả [].
    """
    folder = os.path.dirname(edf_path)
    record_name = os.path.splitext(os.path.basename(edf_path))[0]
    wfdb_record = os.path.join(folder, record_name)

    try:
        ann = wfdb.rdann(wfdb_record, "edf.seizures")
    except Exception:
        return []

    if ann.fs is None or ann.sample is None or len(ann.sample) < 2:
        return []

    fs = float(ann.fs)
    samples = ann.sample

    seizures = []
    for i in range(0, len(samples) - 1, 2):
        start_sec = samples[i] / fs
        end_sec = samples[i + 1] / fs
        if end_sec > start_sec:
            seizures.append((start_sec, end_sec))

    return seizures


# =========================================================
# 3) EEG PREPROCESSING
# =========================================================
def preprocess_raw(
    raw,
    l_freq: float = 0.5,
    h_freq: float = 40.0,
    notch_freq: Optional[float] = 60.0,
    resample_sfreq: Optional[float] = None,
):
    """
    Tiền xử lý EEG cơ bản:
    1) Chọn 18 kênh cần dùng
    2) Notch filter
    3) Bandpass
    4) Resample (nếu muốn)
    """
    raw = pick_required_channels(raw, CH_LABELS)

    sfreq = float(raw.info["sfreq"])

    if notch_freq is not None:
        notch_freqs = []
        f = notch_freq
        while f < sfreq / 2:
            notch_freqs.append(f)
            f += notch_freq

        if len(notch_freqs) > 0:
            raw.notch_filter(freqs=notch_freqs, verbose=False)

    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)

    if resample_sfreq is not None and not np.isclose(resample_sfreq, sfreq):
        raw.resample(resample_sfreq, verbose=False)

    return raw


# =========================================================
# 4) HELPER FUNCTIONS FOR STRICT LABELING
# =========================================================
def overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    """
    Kiểm tra 2 khoảng [a_start, a_end) và [b_start, b_end) có overlap không.
    """
    return (a_start < b_end) and (b_start < a_end)


def contained_in(inner_start: float, inner_end: float, outer_start: float, outer_end: float) -> bool:
    """
    Kiểm tra [inner_start, inner_end) có nằm hoàn toàn trong [outer_start, outer_end) không.
    """
    return (outer_start <= inner_start) and (inner_end <= outer_end)


def valid_interval(start: float, end: float) -> bool:
    return end > start


def infer_patient_id(edf_path: str) -> str:
    """
    Cố gắng suy ra patient_id từ path/file name.
    Ví dụ CHB-MIT: chb01, chb02, ...
    Nếu không match thì lấy tên folder cha.
    """
    path_lower = edf_path.lower()
    m = re.search(r"(chb\d+)", path_lower)
    if m:
        return m.group(1)
    return os.path.basename(os.path.dirname(edf_path))


# =========================================================
# 5) STRICT LABELING RULE
# =========================================================
def get_window_label_strict(
    win_start_t: float,
    win_end_t: float,
    seizures_sec: List[Tuple[float, float]],
    sph_min: float = 0.0,
    sop_min: float = 10.0,
    postictal_min: float = 60.0,
    interictal_gap_min: float = 60.0,
) -> int:
    """
    Gán nhãn strict cho 1 window dựa trên toàn bộ khoảng thời gian [win_start_t, win_end_t).

    Nhãn:
      1  = preictal
      0  = interictal
     -1  = ignore

    Rule strict:
      - preictal chỉ khi toàn bộ window nằm trọn trong vùng preictal
      - nếu window chạm ictal / postictal / buffer / transition / một phần preictal -> ignore
      - còn lại -> interictal

    Các vùng quanh mỗi seizure:
      preictal   : [start - (SPH + SOP), start - SPH)
      buffer     : [start - SPH, start)
      ictal      : [start, end)
      postictal  : [end, end + postictal)
      transition : [start - interictal_gap, start - (SPH + SOP))
    """
    if not seizures_sec:
        return 0

    sph = sph_min * 60.0
    sop = sop_min * 60.0
    post = postictal_min * 60.0
    gap = interictal_gap_min * 60.0

    preictal_len = sph + sop
    is_full_preictal = False

    for start, end in seizures_sec:
        pre_start = start - preictal_len
        pre_end = start - sph

        buffer_start = start - sph
        buffer_end = start

        ictal_start = start
        ictal_end = end

        post_start = end
        post_end = end + post

        transition_start = start - gap
        transition_end = pre_start

        if overlaps(win_start_t, win_end_t, ictal_start, ictal_end):
            return -1

        if valid_interval(post_start, post_end) and overlaps(win_start_t, win_end_t, post_start, post_end):
            return -1

        if valid_interval(buffer_start, buffer_end) and overlaps(win_start_t, win_end_t, buffer_start, buffer_end):
            return -1

        if valid_interval(transition_start, transition_end) and overlaps(win_start_t, win_end_t, transition_start, transition_end):
            return -1

        if valid_interval(pre_start, pre_end) and overlaps(win_start_t, win_end_t, pre_start, pre_end):
            if contained_in(win_start_t, win_end_t, pre_start, pre_end):
                is_full_preictal = True
            else:
                return -1

    if is_full_preictal:
        return 1

    return 0


# =========================================================
# 6) CUT WINDOWS + LABEL ONE EDF
# =========================================================
def label_one_file(
    edf_path: str,
    seizures_sec: List[Tuple[float, float]],
    window_size_sec: float = 8.0,
    preictal_step_sec: float = 4.0,
    interictal_step_sec: float = 60.0,
    ignore_step_sec: Optional[float] = None,
    sph_min: float = 0.0,
    sop_min: float = 10.0,
    postictal_min: float = 60.0,
    interictal_gap_min: float = 60.0,
    l_freq: float = 0.5,
    h_freq: float = 40.0,
    notch_freq: Optional[float] = 60.0,
    resample_sfreq: Optional[float] = 128.0,
):
    """
    Đọc 1 file EDF -> preprocess -> cắt window -> gán nhãn strict.

    Ý nghĩa step:
    - preictal_step_sec   : bước trượt cho cửa sổ preictal
    - interictal_step_sec : bước trượt cho cửa sổ interictal
    - ignore_step_sec     : bước trượt khi window bị ignore
                            nếu None -> dùng interictal_step_sec

    Trả về:
      X                : shape (N, C, T)
      y                : shape (N,)
      sfreq            : sampling rate sau preprocess
      win_start_secs   : shape (N,)
      win_end_secs     : shape (N,)
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Channel names are not unique, found duplicates.*",
            category=RuntimeWarning,
        )
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    raw = preprocess_raw(
        raw,
        l_freq=l_freq,
        h_freq=h_freq,
        notch_freq=notch_freq,
        resample_sfreq=resample_sfreq,
    )

    sfreq = float(raw.info["sfreq"])
    data = np.asarray(raw.get_data(), dtype=np.float32)   # (C, T)
    n_channels, n_samples = data.shape

    if ignore_step_sec is None:
        ignore_step_sec = interictal_step_sec

    win = int(round(window_size_sec * sfreq))
    step_preictal = int(round(preictal_step_sec * sfreq))
    step_interictal = int(round(interictal_step_sec * sfreq))
    step_ignore = int(round(ignore_step_sec * sfreq))

    if win <= 0:
        raise ValueError("window_size_sec phải > 0")
    if step_preictal <= 0:
        raise ValueError("preictal_step_sec phải > 0")
    if step_interictal <= 0:
        raise ValueError("interictal_step_sec phải > 0")
    if step_ignore <= 0:
        raise ValueError("ignore_step_sec phải > 0")

    if n_samples < win:
        return (
            np.empty((0, n_channels, win), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            sfreq,
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    X, y = [], []
    win_start_secs, win_end_secs = [], []

    total_windows = 0
    ignored_windows = 0
    preictal_windows = 0
    interictal_windows = 0

    s = 0
    while s <= n_samples - win:
        total_windows += 1
        e = s + win

        cur_win_start_t = s / sfreq
        cur_win_end_t = e / sfreq

        label = get_window_label_strict(
            win_start_t=cur_win_start_t,
            win_end_t=cur_win_end_t,
            seizures_sec=seizures_sec,
            sph_min=sph_min,
            sop_min=sop_min,
            postictal_min=postictal_min,
            interictal_gap_min=interictal_gap_min,
        )

        if label == -1:
            ignored_windows += 1
            s += step_ignore
            continue

        window = data[:, s:e]

        X.append(window)
        y.append(label)
        win_start_secs.append(cur_win_start_t)
        win_end_secs.append(cur_win_end_t)

        if label == 1:
            preictal_windows += 1
            s += step_preictal
        else:
            interictal_windows += 1
            s += step_interictal

    print(
        f"[{os.path.basename(edf_path)}] "
        f"total={total_windows}, kept={len(y)}, "
        f"ignore={ignored_windows}, preictal={preictal_windows}, interictal={interictal_windows}"
    )

    if len(X) == 0:
        return (
            np.empty((0, n_channels, win), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            sfreq,
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    win_start_secs = np.asarray(win_start_secs, dtype=np.float32)
    win_end_secs = np.asarray(win_end_secs, dtype=np.float32)

    return X, y, sfreq, win_start_secs, win_end_secs


# =========================================================
# 7) SAVE NPZ WITH METADATA
# =========================================================
def save_npz_like_input(
    edf_path: str,
    input_root: str,
    output_root: str,
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    win_start_secs: np.ndarray,
    win_end_secs: np.ndarray,
):
    """
    Ví dụ:
      input_root/.../chb03/chb03_01.edf
      -> output_root/.../chb03/chb03_01.npz
    """
    rel_path = os.path.relpath(edf_path, input_root)
    rel_no_ext = os.path.splitext(rel_path)[0]
    out_path = os.path.join(output_root, rel_no_ext + ".npz")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    patient_id = infer_patient_id(edf_path)

    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        sfreq=np.float32(sfreq),
        window_start_sec=win_start_secs,
        window_end_sec=win_end_secs,
        source_edf=np.array(edf_path),
        patient_id=np.array(patient_id),
    )

    return out_path


# =========================================================
# 8) PROCESS A WHOLE FOLDER
# =========================================================
def process_folder(
    input_root: str,
    output_root: str,
    pattern: str = "**/*.edf",
    window_size_sec: float = 8.0,
    preictal_step_sec: float = 4.0,
    interictal_step_sec: float = 60.0,
    ignore_step_sec: Optional[float] = None,
    sph_min: float = 0.0,
    sop_min: float = 10.0,
    postictal_min: float = 60.0,
    interictal_gap_min: float = 60.0,
    l_freq: float = 0.5,
    h_freq: float = 40.0,
    notch_freq: Optional[float] = 60.0,
    resample_sfreq: Optional[float] = 128.0,
):
    """
    Xử lý toàn bộ folder EDF.
    Với mỗi file:
      - đọc seizure annotation
      - cắt window + gán nhãn strict
      - lưu .npz sang output folder
    """
    edf_files = sorted(glob.glob(os.path.join(input_root, pattern), recursive=True))

    if len(edf_files) == 0:
        print(f"No EDF files found in: {input_root}")
        return

    print(f"Found {len(edf_files)} EDF files.")

    total_files = 0
    total_windows = 0
    total_preictal = 0
    total_interictal = 0

    for edf_path in tqdm(edf_files, desc="Processing EDF", unit="file"):
        try:
            seizures_sec = read_seizures_sec(edf_path)

            X, y, sfreq, win_start_secs, win_end_secs = label_one_file(
                edf_path=edf_path,
                seizures_sec=seizures_sec,
                window_size_sec=window_size_sec,
                preictal_step_sec=preictal_step_sec,
                interictal_step_sec=interictal_step_sec,
                ignore_step_sec=ignore_step_sec,
                sph_min=sph_min,
                sop_min=sop_min,
                postictal_min=postictal_min,
                interictal_gap_min=interictal_gap_min,
                l_freq=l_freq,
                h_freq=h_freq,
                notch_freq=notch_freq,
                resample_sfreq=resample_sfreq,
            )

            out_path = save_npz_like_input(
                edf_path=edf_path,
                input_root=input_root,
                output_root=output_root,
                X=X,
                y=y,
                sfreq=sfreq,
                win_start_secs=win_start_secs,
                win_end_secs=win_end_secs,
            )

            n_interictal = int((y == 0).sum()) if len(y) > 0 else 0
            n_preictal = int((y == 1).sum()) if len(y) > 0 else 0

            total_files += 1
            total_windows += len(y)
            total_interictal += n_interictal
            total_preictal += n_preictal

            tqdm.write(
                f"OK   {os.path.basename(edf_path)} "
                f"| saved={out_path} "
                f"| windows={len(y)} "
                f"| interictal={n_interictal} "
                f"| preictal={n_preictal}"
            )

        except Exception as e:
            tqdm.write(f"SKIP {os.path.basename(edf_path)} | {e}")

    print("\n========== SUMMARY ==========")
    print(f"Processed files : {total_files}")
    print(f"Total windows   : {total_windows}")
    print(f"Interictal      : {total_interictal}")
    print(f"Preictal        : {total_preictal}")


# =========================================================
# 9) EXAMPLE USAGE
# =========================================================
if __name__ == "__main__":
    INPUT_ROOT  = r"C:\Users\Admin\Documents\Thesis\CHB-MIT"
    OUTPUT_ROOT = r"C:\Users\Admin\Documents\Thesis\chbmit-30min"

    process_folder(
        input_root=INPUT_ROOT,
        output_root=OUTPUT_ROOT,

        # Sliding window
        window_size_sec=8.0,
        preictal_step_sec=4.0,       # preictal lấy dày
        interictal_step_sec=20.0,    # interictal lấy thưa, đổi số này tùy ý
        ignore_step_sec=60.0,        # có thể để None để tự dùng interictal_step_sec

        # Prediction setup
        sph_min=0.0,
        sop_min=30.0,                 # preictal = 30 phút trước cơn
        postictal_min=60.0,
        interictal_gap_min=30.0,

        # EEG filtering
        l_freq=0.5,
        h_freq=40.0,
        notch_freq=60.0,
        resample_sfreq=256.0
    )
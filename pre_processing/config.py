from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    mel_filter_size: int  # Number of mel filter banks. (Default: 80)
    sampling_rate: int  # # Sampling rate of raw audio. (Default: 16000)
    win_time: float  # Window size (sec). (Default: 0.025)
    stride_time: float  # Length of hop between STFT windows (sec). (Default: 0.01)
    n_fft: int  # Size of FFT. (Default: 2048)
    fixed_time: float  # 15s (sampling rate 16000)

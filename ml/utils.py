import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


exts1 = ["ascii", "cm1", "cp1", "ch3", "cc1", "cn1", "cc3", "cu4"]
exts2 = ["cn4", "cn3", "cc2", "cs1", "REC", "txt"]


def rename_dataset():
    for ext in exts1 + exts2:
        os.makedirs(f"dataset/{ext}", exist_ok=True)
        for Np in range(1, 41):
            for Nr in range(1, 3):
                for ch in range(1, 3):
                    try:
                        move(
                            f"dataset/dataset_O{ch}/Np {Np}/Nr {Nr}/N-{Nr}.{ext}",
                            f"dataset/{ext}/{Np}-{Nr}-O{ch}.{ext}",
                        )
                    except:
                        pass
                    

def read_signal(np: int, nr: int, ch: int) -> list[int]:
    assert 1 <= np <= 40
    assert 1 <= nr <= 2
    assert 1 <= ch <= 2

    with open(f"dataset/ascii/{np}-{nr}-O{ch}.ascii", "r") as signal_file:
        signal = []
        for line in signal_file:
            signal.append(int(line))

    return signal


def visualize_signal(signal, freq=1):
    fig, ax = plt.subplots()
    x = np.linspace(0, len(signal) - 1, len(signal)) / freq
    ax.plot(x, signal)
    

def get_rhythms_matrix():
    columns = ["delta", "theta", "alpha", "beta", "gamma"]
    rhythms = []
    for pat in range(1, 41):
        for rec in range(1, 3):
            rhythms.append(
                get_rhythms(fir_filter(read_signal(pat, rec, 1), FILT_COEF))
                + get_rhythms(fir_filter(read_signal(pat, rec, 2), FILT_COEF))
            )
            

def rhythms_matrix_to_df(rhythms):
    df = pd.DataFrame(
        rhythms,
        columns=[f"{col}_O{i}" for col in columns for i in range(1, 3)],
    )
    df["pat"] = [pat for pat in range(1, 41) for rec in range(1, 3)]
    df["rec"] = [rec for pat in range(1, 41) for rec in range(1, 3)]
    df["apnoe"] = [
        0 if pat in healthy_indices else 1 for pat in range(1, 41) for rec in range(1, 3)
    ]

    cols = list(df.columns)
    df = df[cols[10:12] + cols[:10] + cols[12:]]
    return df

"""Сохранение и визуализация метрик (по образцу опорного примера)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class MetricLogger:
    def __init__(self):
        self.df = pd.DataFrame(
            {"metric": pd.Series(dtype="str"),
             "alg": pd.Series(dtype="str"),
             "value": pd.Series(dtype="float"),
             "stage": pd.Series(dtype="str")}
        )

    def add(self, metric: str, alg: str, value: float, stage: str = "baseline"):
        mask = (
            (self.df["metric"] == metric)
            & (self.df["alg"] == alg)
            & (self.df["stage"] == stage)
        )
        self.df = self.df[~mask]
        self.df = pd.concat(
            [self.df, pd.DataFrame([{
                "metric": metric, "alg": alg, "value": value, "stage": stage
            }])],
            ignore_index=True,
        )

    def get_data_for_metric(self, metric: str, stage: str = "baseline", ascending=True):
        temp = self.df[(self.df["metric"] == metric) & (self.df["stage"] == stage)]
        temp = temp.sort_values(by="value", ascending=ascending)
        return temp["alg"].values, temp["value"].values

    def plot(self, title: str, metric: str, stage: str = "baseline",
             ascending=True, figsize=(7, 5), save_path=None):
        labels, values = self.get_data_for_metric(metric, stage, ascending)
        fig, ax = plt.subplots(figsize=figsize)
        pos = np.arange(len(values))
        ax.barh(pos, values, align="center", height=0.5, tick_label=labels)
        ax.set_title(title)
        for p, v in zip(pos, values):
            ax.text(v * 0.5 if v > 0 else 0.01, p, f"{v:.3f}", va="center", color="white", fontweight="bold")
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        return fig

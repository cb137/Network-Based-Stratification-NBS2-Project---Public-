import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(
    history,
    M_train,
    M_val=None,
    save_path=None,
    title_prefix="NBS²",
    figsize=(10, 10)
):
    """
    Make 2 stacked subplots:
        Subplot 1 → Cost-per-tumor (train + val)
        Subplot 2 → Accuracy (train + val)
    """

    # Extract data
    epochs = np.array([h["epoch"] for h in history])
    cost_train_raw = np.array([h["wmw_cost"] for h in history])
    acc_train = np.array([h["acc_train"] for h in history])

    # Convert to cost-per-tumor
    cost_train = cost_train_raw / M_train

    # Validation
    has_val = "wmw_cost_val" in history[0]
    if has_val:
        cost_val_raw = np.array([h["wmw_cost_val"] for h in history])
        acc_val = np.array([h["acc_val"] for h in history])
        cost_val = cost_val_raw / M_val

    # ------------------------------
    # Create 2 subplots
    # ------------------------------
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    ax_cost, ax_acc = axes

    # ============================================================
    # Subplot 1: Cost-per-tumor curves
    # ============================================================
    ax_cost.plot(
        epochs, cost_train,
        label="Cost per tumor (Train)",
        color="tab:blue", linewidth=2
    )

    if has_val:
        ax_cost.plot(
            epochs, cost_val,
            label="Cost per tumor (Val)",
            color="tab:orange", linewidth=2
        )

    ax_cost.set_ylabel("Cost per tumor", fontsize=14)
    ax_cost.set_title(f"{title_prefix} – Cost per Tumor", fontsize=16)
    ax_cost.grid(alpha=0.3)
    ax_cost.legend(fontsize=12)

    # ============================================================
    # Subplot 2: Accuracy curves
    # ============================================================
    ax_acc.plot(
        epochs, acc_train,
        label="Accuracy (Train)",
        color="tab:blue", linewidth=2
    )

    if has_val:
        ax_acc.plot(
            epochs, acc_val,
            label="Accuracy (Val)",
            color="tab:orange", linewidth=2
        )

    ax_acc.set_xlabel("Iterations", fontsize=14)
    ax_acc.set_ylabel("Accuracy", fontsize=14)
    ax_acc.set_title(f"{title_prefix} – Accuracy", fontsize=16)
    ax_acc.grid(alpha=0.3)
    ax_acc.legend(fontsize=12)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"[Saved plot] {save_path}")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def calculate_expert_usage(expert_ids):
    """
    expert_ids: list tên expert được gọi tương ứng với mỗi ảnh
    """
    counts = Counter(expert_ids)
    total = sum(counts.values())
    usage = {}
    for expert, count in counts.items():
        tmp = count/total
        usage[expert] = round(tmp, 3)
    return usage

def compare_confusion_matrices(y_true, y_pred_baseline, y_pred_moe, labels):
    cm_base = confusion_matrix(y_true, y_pred_baseline, labels=labels)
    cm_moe = confusion_matrix(y_true, y_pred_moe, labels=labels)

    diff = cm_base - cm_moe

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    sns.heatmap(cm_base, annot=True, fmt="d", ax=ax[0], cmap="Blues")
    ax[0].set_title("Baseline Confusion Matrix")

    sns.heatmap(cm_moe, annot=True, fmt="d", ax=ax[1], cmap="Greens")
    ax[1].set_title("MoE Confusion Matrix")

    sns.heatmap(diff, annot=True, fmt="d", center=0, ax=ax[2], cmap="coolwarm")
    ax[2].set_title("Difference (Baseline - MoE)")

    plt.show()

def evaluate_moe(y_true, y_pred, y_pred_baseline,
                 y_true_fungal, y_pred_fungal,
                 y_true_bv, y_pred_bv,
                 y_true_phys, y_pred_phys,
                 expert_ids, all_labels):
    metrics = {
        'overall_accuracy': accuracy_score(y_true, y_pred),
        'per_group_f1': {
            'fungal': f1_score(y_true_fungal, y_pred_fungal, average='macro'),
            'bacterial_viral': f1_score(y_true_bv, y_pred_bv, average='macro'),
            'physiological': f1_score(y_true_phys, y_pred_phys, average='macro'),
        },
        'expert_utilization': calculate_expert_usage(expert_ids),
        'confusion_reduction': compare_confusion_matrices(y_true, y_pred_baseline, y_pred, labels=all_labels)
    }
    return metrics
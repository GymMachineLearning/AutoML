import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score, accuracy_score

def plot_statistics_per_class(statistics, labels_type):
    """
    Rysuje wykres przedstawiający statystyki dla każdej klasy.

    Args:
        labels (list): Lista nazw klas.
        statistics (dict): Słownik statystyk z kluczami 'precision', 'recall', 'f1', 'auc', zawierający listy wyników.
    """
    x = np.arange(len(labels_type))  # Pozycje na osi X

    width = 0.2  # Szerokość słupka
    fig, ax = plt.subplots(figsize=(10, 6))

    # Rysowanie słupków dla każdej metryki
    ax.bar(x - width * 1.5, statistics['precision'], width, label='Precision', color='skyblue')
    ax.bar(x - width * 0.5, statistics['recall'], width, label='Recall', color='lightgreen')
    ax.bar(x + width * 0.5, statistics['f1'], width, label='F1 Score', color='salmon')
    ax.bar(x + width * 1.5, statistics['auc'], width, label='AUC', color='orange')

    # Ustawienia osi i etykiet
    ax.set_xlabel('Classes', fontsize=12)
    ax.set_ylabel('Scores', fontsize=12)
    ax.set_title('Statistics per Class', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_type, rotation=90, ha='right')
    ax.legend()

    # Dodanie wartości nad słupkami
    for i, label in enumerate(labels_type):
        ax.text(i - width * 1.5, statistics['precision'][i] + 0.02, f"{statistics['precision'][i]:.2f}", ha='center', fontsize=8)
        ax.text(i - width * 0.5, statistics['recall'][i] + 0.02, f"{statistics['recall'][i]:.2f}", ha='center', fontsize=8)
        ax.text(i + width * 0.5, statistics['f1'][i] + 0.02, f"{statistics['f1'][i]:.2f}", ha='center', fontsize=8)
        ax.text(i + width * 1.5, statistics['auc'][i] + 0.02, f"{statistics['auc'][i]:.2f}", ha='center', fontsize=8)

    plt.tight_layout()
    plt.show()


def compute_and_plot_statistics(y_test_flat, y_pred_flat, labels_type, print_stats=False, plot_stats=False):
        """
        Oblicza statystyki dla każdej klasy i tworzy wykres.
    
        Args:
            y_test_flat (np.ndarray): Spłaszczona macierz testowa (n_samples, n_classes).
            y_pred_flat (np.ndarray): Spłaszczona macierz przewidywań (n_samples, n_classes).
            labels_type (list): Lista nazw klas.
        """
        precision_scores = []
        recall_scores = []
        f1_scores = []
        auc_scores = []
        accuracy_scores = []
    
        for i in range(len(labels_type)):
            y_test_ = y_test_flat[:, i]
            y_pred_ = y_pred_flat[:, i]
    
            precision = precision_score(y_test_, y_pred_, zero_division=0)
            recall = recall_score(y_test_, y_pred_, zero_division=0)
            f1 = f1_score(y_test_, y_pred_, zero_division=0)
            accuracy = accuracy_score(y_test_, y_pred_)
            
            if sum(y_test_) != 0:
                auc = roc_auc_score(y_test_, y_pred_)
            else:
                auc = 0  # Brak danych do obliczenia AUC
    
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            auc_scores.append(auc)
            accuracy_scores.append(accuracy)
            
            if print_stats == True:
                print(f"Class {labels_type[i]} ({i})  statistics:")
                print("   Precision:", precision)
                print("   Recall:", recall)
                print("   F1 Score:", f1)
                print("   Accuracy:", accuracy)
                print(f"   AUC for class {i}:", auc)
    
        statistics = {
            'precision': precision_scores,
            'recall': recall_scores,
            'f1': f1_scores,
            'auc': auc_scores,
            'accuracy': accuracy_scores
        }
        
        if plot_stats == True:
            plot_statistics_per_class(statistics, labels_type)
            
        return statistics
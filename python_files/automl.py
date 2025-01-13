import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from python_files.Preprocess import PaddingEstimator, add_pad, extract_features_with_window, process_labels_with_window, WindowFeatureExtractor, WindowLabelProcessor, process_labels_with_window_2d, PCADimensionReducer


from python_files.Plot import compute_and_plot_statistics, plot_statistics_per_class, plot_data_raport

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
import xgboost as xgb
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

class CustomPipeline(Pipeline):
    def fit(self, X, y=None):
        # Wypisanie kształtów przed transformacją
        print(f'Original X shape: {X[0].shape}')
        print(f'Original y shape: {y[0].shape}')
        
        # Wywołanie oryginalnej metody fit
        super().fit(X, y)
        
        # Wypisanie kształtów po transformacji
        for step_name, step_transformer in self.steps:
            if hasattr(step_transformer, 'transform'):
                X, y = step_transformer.transform(X, y)  # Zastosowanie transformacji
                print(f'Post-transformation X shape: {X[0].shape}')
                print(f'Post-transformation y shape: {y[0].shape}')
        print(f'X type: {type(X)}')
        print(f'y type: {type(y)}')
        return self

from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector

def debug_function(X):
    # print(f"Shape after transformation: {X.shape}")
    return X


class AutoMlMultiLabelClassifier:
    def __init__(self, model=None, window_size=120, step_size=120, labels_type=None):
        """

        Args:
            model (sklearn.base.BaseEstimator, optional): Model klasyfikacyjny. 
                Domyślnie RandomForestClassifier.
        """
        
        self.model = model if model else RandomForestClassifier()
        self.is_fitted = False
        self.window_size = window_size # Rozmiar okna dla obrabiania X
        self.step_size = step_size # Rozmiar okna dla obrabiania y
        self.labels_type = labels_type 
        
    def trimming_data(self, X, y):
        # Trimming z wykorzstyaniem funkcji pojedyńczych
        X = self.trimming_data_y(X)
        y = self.trimming_data_y(y)
        return X, y

    def trimming_data_y(self, y):
        # Znajdź minimalną długość
        min_length = min(arr.shape[0] for arr in y)
        
        trimmed_segments_y = [
            arr[i:i+min_length, :] 
            for arr in y 
            for i in range(0, arr.shape[0] - min_length + 1, min_length)
        ]
        
        y = np.array(trimmed_segments_y)

        y = y.reshape(-1, y.shape[2])

        y = pd.DataFrame(y)

        y.columns = ['label_' + str(col) for col in y.columns]

        return y

    def trimming_data_X(self, X):
        # Znajdź minimalną długość
        min_length = min(arr.shape[0] for arr in X)

        # Wyodrębnienie maksymalnej liczby segmentów o długości min_length
        trimmed_segments = [
            arr[i:i+min_length, :] 
            for arr in X 
            for i in range(0, arr.shape[0] - min_length + 1, min_length)
        ]
        
        X = np.array(trimmed_segments)

        X = X.reshape(-1, X.shape[2])

        X = pd.DataFrame(X)

        X.columns = ['feature_' + str(col) for col in X.columns]

        return X

    def fit(self, X, y):
        """
        Automatycznie trenuje i optymalizuje modele ML na podanych danych.

        Args:
            X (np.ndarray): Dane wejściowe (features).
            y (np.ndarray): Etykiety (labels).
        """
        try:
            # Definicja parametrów dla GridSearchCV
            param_distributions = {
                'RandomForest': {
                    'model__n_estimators': [100, 200, 300, 500, 1000],
                    # 'model__max_depth': [10, 20, None],
                    # 'model__min_samples_split': [2, 5, 10],
                },
                'XGBoost': {
                    'model__estimator__n_estimators': [100, 200, 500, 1000],
                    # 'model__estimator__max_depth': [3, 6, 10],
                    # 'model__estimator__learning_rate': [0.01, 0.1, 0.3],
                    # 'model__estimator__subsample': [0.8, 1.0],
                }
            }

            # Trimming danych
            X, y = self.trimming_data(X, y)
            
            y_test = process_labels_with_window_2d(y, self.window_size, self.step_size)

            # Podział na dane treningowe i testowe
            # pca = PCA(n_components=50)
            # X = pca.fit_transform(X)
            feature_pipeline = Pipeline([
                ('debug1', FunctionTransformer(debug_function, validate=False)),
                ('pca', PCADimensionReducer()),
                ('debug3', FunctionTransformer(debug_function, validate=False)),
            ])
            
            # Tworzymy pipeline dla etykiet (y)
            label_pipeline = Pipeline([
                ('debug11', FunctionTransformer(debug_function, validate=False)),
                ('label_processing', WindowLabelProcessor(window_size=self.window_size, step=self.step_size)),
                ('debug21', FunctionTransformer(debug_function, validate=False)),

            ])
            
            # # Łączymy oba pipeline'y w jeden
            # from sklearn.compose import ColumnTransformer
            # full_pipeline = ColumnTransformer([
            #     ('features', feature_pipeline, make_column_selector(pattern='^feature_')),  # Przetwarzanie cech
            # ])
            
            # Tworzenie pipeline'ów dla każdego modelu
            pipelines = {
                'RandomForest': Pipeline([
                    ('preprocessing', feature_pipeline),
                    ('model', RandomForestClassifier())
                ]),
                'XGBoost': Pipeline([
                    ('preprocessing', feature_pipeline),
                    ('model', OneVsRestClassifier(XGBClassifier(n_jobs=-1, eval_metric='auc',
                                                                objective='binary:hinge', tree_method='hist')))
                ])
            }

            ext = WindowLabelProcessor(window_size=self.window_size, step=self.step_size)
            y = ext.transform(pd.DataFrame(y))
            
            ext = WindowFeatureExtractor(window_size=self.window_size, step_size=self.step_size)
            X = ext.transform(pd.DataFrame(X))
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            print('fituje')
            pipelines['RandomForest'].fit(X_train, y_train)
            best_model = None
            best_score = 0
            best_params = {}

            for name, pipeline in pipelines.items():
                print(f"Trenuję model: {name}")

                # Inicjalizacja GridSearchCV
                grid_search = GridSearchCV(pipeline, param_distributions[name],
                                           cv=2, scoring='accuracy', n_jobs=-1, error_score='raise')
                grid_search.fit(X_train, y_train)

                # Najlepsze wyniki dla danego modelu
                print(f"{name} - Best Parameters: {grid_search.best_params_}")
                print(f"{name} - Best Cross-Validation Score: {grid_search.best_score_}")

                # Sprawdzanie, czy to najlepszy model
                if grid_search.best_score_ > best_score:
                    best_model = grid_search.best_estimator_
                    best_score = grid_search.best_score_
                    best_params = grid_search.best_params_

            # Predykcja i ocena najlepszego modelu
            print("\nNajlepszy model:", best_model)
            print("Najlepsze parametry:", best_params)

            y_pred = best_model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred, average='weighted')

            # Wyćwiczyliśmy model
            self.is_fitted = True

            # Zapis modelu
            self.model = best_model
            
            print(f"Accuracy: {accuracy}")
            print(f"Recall: {recall}")

        except Exception as e:
            print(f"Wystąpił błąd podczas treningu: {e}")
            
    def predict(self, X):
        """
        Przewiduje etykiety dla podanych danych.

        Args:
            X (np.ndarray): Dane wejściowe (features).

        Returns:
            np.ndarray: Przewidywane etykiety.
        """
        if not self.is_fitted:
            raise ValueError("Model nie został jeszcze wytrenowany. Użyj metody fit przed predict.")
        try:
            # Triming danych
            X = self.trimming_data_X(X)
            
            ext = WindowFeatureExtractor(window_size=self.window_size, step_size=self.step_size)
            X = ext.transform(pd.DataFrame(X))

            predictions = self.model.predict(X)
            
            return predictions
        except Exception as e:
            print(f"Błąd podczas przewidywania: {e}")
            return None

    def score(self, X, y, plot_stats=False):
        """
        Oblicza dokładność modelu na podanych danych testowych.

        Args:
            X (np.ndarray): Dane wejściowe (features).
            y (np.ndarray): Rzeczywiste etykiety (labels).

        Returns:
            dictinary: statistics - słownik zawierający informację o wszystkich najważniejszych statystykach, dla każdej klasy.

       statistics = {
            'precision': precision_scores,
            'recall': recall_scores,
            'f1': f1_scores,
            'auc': auc_scores,
            'accuracy': accuracy_scores
        }
        """

        
        if not self.is_fitted:
            raise ValueError("Model nie został jeszcze wytrenowany. Użyj metody fit przed score.")
        try:
            y_pred = self.predict(X)
            
            # Trimming danych
            y = self.trimming_data_y(y)

            y_test = process_labels_with_window_2d(y, self.window_size, self.step_size)

            statistics = compute_and_plot_statistics(y_test, y_pred, self.labels_type, print_stats=False, plot_stats=plot_stats)
            
            return statistics
        except Exception as e:
            print(f"Błąd podczas obliczania dokładności: {e}")
            return None

    def raport_scores(self, statistics):
        """
        Funkcja wyświetla najważniejsze statystki dla modelu. Tworzy wyrkesu i podsumowania. Szczególnie liczy efektywnośc modelu dla każdej klasy oddzielnie.

        Args:
            dictionary: statistics - słownik zawierający informację o wszystkich najważniejszych statystykach, dla każdej klasy.

       statistics = {
            'precision': precision_scores,
            'recall': recall_scores,
            'f1': f1_scores,
            'auc': auc_scores,
            'accuracy': accuracy_scores
        }
        """
        print("Wykres przedstawiający statystyki (precision, recall, f1, AUC oraz accuracy) dla każdej klasy błędów z osobna")
        # Wykresy dla statystyk dla różnych klas.
        plot_statistics_per_class(statistics, self.labels_type)

        # Poniżej wypisujemy wyniki w postaci liczbowej
        statistics_list = sorted(statistics)  # Sortujemy metryki alfabetycznie
        statistics_dict = {metric: statistics[metric] for metric in statistics_list}
        
        # Tworzenie DataFrame
        df = pd.DataFrame.from_dict(statistics_dict, orient="index")
        df.columns = [f"Value {self.labels_type[i]}" for i in range(df.shape[1])]  # Nazwy kolumn
        
        # Wyświetlenie macierzy
        print("Wyniki w postaci macierzy")
        print(df)


    def raport_data(self, X, y):
        """
        Funkcja wyświetla najważniejsze statystki i informacje na temat zestawu danych.

        Args:
            X (np.ndarray): Dane wejściowe (features).
            y (np.ndarray): Rzeczywiste etykiety (labels).

        """
        # Trimming danych
        X, y = self.trimming_data(X, y)
            
        plot_data_raport(X, y, self.labels_type)
        

# Przykład użycia
if __name__ == "__main__":
    # Generowanie przykładowych danych
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Użycie klasy
    cls = AutoSklearnClassifier()
    cls.fit(X_train, y_train)
    predictions = cls.predict(X_test)
    accuracy = cls.score(X_test, y_test)

    print("Przewidywania:", predictions[:10])
    print("Dokładność:", accuracy)

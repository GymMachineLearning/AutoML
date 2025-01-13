import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from Preprocess import PaddingEstimator, add_pad, extract_features_with_window, process_labels_with_window, WindowFeatureExtractor, WindowLabelProcessor, process_labels_with_window_2d, PCADimensionReducer

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
    print(f"Shape after transformation: {X.shape}")
    return X


class AutoMlMultiLabelClassifier:
    def __init__(self, model=None):
        """

        Args:
            model (sklearn.base.BaseEstimator, optional): Model klasyfikacyjny. 
                Domyślnie RandomForestClassifier.
        """
        self.model = model if model else RandomForestClassifier()
        self.is_fitted = False
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
                        # Znajdź minimalną długość
            min_length = min(arr.shape[0] for arr in X)
            
            # Tworzymy pipeline dla cech (X)
            

            
            # Wyodrębnienie maksymalnej liczby segmentów o długości min_length
            trimmed_segments = [
                arr[i:i+min_length, :] 
                for arr in X 
                for i in range(0, arr.shape[0] - min_length + 1, min_length)
            ]
            
            # # Wynik
            # print(f"Liczba segmentów: {len(trimmed_segments)}")
            # for i, segment in enumerate(trimmed_segments):
            #     print(f"Segment {i+1}: {segment.shape}")


            trimmed_segments_y = [
                arr[i:i+min_length, :] 
                for arr in y 
                for i in range(0, arr.shape[0] - min_length + 1, min_length)
            ]
            
            # # Wynik
            # print(f"Liczba segmentów y: {len(trimmed_segments_y)}")
            # for i, segment in enumerate(trimmed_segments_y):
            #     print(f"Segment {i+1}: {segment.shape}") 
            X = np.array(trimmed_segments)
            y = np.array(trimmed_segments_y)

            X = X.reshape(-1, X.shape[2])
            y = y.reshape(-1, y.shape[2])

            X = pd.DataFrame(X)
            y = pd.DataFrame(y)

            X.columns = ['feature_' + str(col) for col in X.columns]
            y.columns = ['label_' + str(col) for col in y.columns]

            print(X.dtypes)
            print(y.dtypes)
            y_test = process_labels_with_window_2d(y,120, 120)

            print("yacalyt.shape: ",y_test.shape)
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
                ('label_processing', WindowLabelProcessor(window_size=120, step=120)),
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


            ext = WindowLabelProcessor(window_size=120, step=120)
            y = ext.transform( pd.DataFrame(y))
            
            ext = WindowFeatureExtractor(window_size=120, step_size=120)
            X = ext.transform( pd.DataFrame(X))
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

            print(f"Accuracy: {accuracy}")
            print(f"Recall: {recall}")

        except Exception as e:
            print(f"Wystąpił błąd podczas treningu: {e}")
            
    # def fit(self, X, y):
    #     """
    #     Automatycznie trenuje i optymalizuje modele ML na podanych danych.

    #     Args:
    #         X (np.ndarray): Dane wejściowe (features).
    #         y (np.ndarray): Etykiety (labels).
    #     """
    #     try:
    #         # Definicja modeli bazowych
    #         base_models = [
    #             ('RandomForest', RandomForestClassifier()),
    #             ('XGBoost', OneVsRestClassifier(XGBClassifier(n_jobs=8, eval_metric='auc', 
    #                                                           objective='binary:hinge', tree_method='hist')))
    #         ]

    #         # Definicja parametrów dla GridSearchCV
    #         param_grid = {
    #             'RandomForest': {
    #                 'n_estimators': [100, 200, 300],
    #                 'max_depth': [10, 20, 30]
    #             },
    #             'XGBoost': {
    #                 'estimator__n_estimators': [500, 1000, 1500],
    #                 'estimator__max_depth': [10, 20, 30]
    #             }
    #         }

    #         # Tworzenie pipeline
    #         one_rep = 50
    #         division = 1
    #         pipeline = Pipeline([
    #             ('feature_extraction', WindowFeatureExtractor(window_size=one_rep, step_size=one_rep // division)),
    #             ('label_processing', WindowLabelProcessor(window_size=one_rep, step=one_rep // division)),
    #         ])

    #         # Przekształcanie danych
    #         feature_extractor = pipeline.named_steps['feature_extraction']
    #         label_processor = pipeline.named_steps['label_processing']
    #         for x in X:
    #             print(x.shape)
               
    #         # Znajdź minimalną długość
    #         min_length = min(arr.shape[0] for arr in X)
            
    #         # Wyodrębnienie maksymalnej liczby segmentów o długości min_length
    #         trimmed_segments = [
    #             arr[i:i+min_length, :] 
    #             for arr in X 
    #             for i in range(0, arr.shape[0] - min_length + 1, min_length)
    #         ]
            
    #         # # Wynik
    #         # print(f"Liczba segmentów: {len(trimmed_segments)}")
    #         # for i, segment in enumerate(trimmed_segments):
    #         #     print(f"Segment {i+1}: {segment.shape}")


    #         trimmed_segments_y = [
    #             arr[i:i+min_length, :] 
    #             for arr in y 
    #             for i in range(0, arr.shape[0] - min_length + 1, min_length)
    #         ]
            
    #         # # Wynik
    #         # print(f"Liczba segmentów y: {len(trimmed_segments_y)}")
    #         # for i, segment in enumerate(trimmed_segments_y):
    #         #     print(f"Segment {i+1}: {segment.shape}") 

            
    #         # X = feature_extractor.transform(X)
    #         # y = label_processor.transform(y)
    #         # y = y[:, [0, 1, 2, 3, 5, 7]]

    #         # # Redukcja wymiarowości
    #         # pca = PCA(n_components=50)
    #         # X = pca.fit_transform(X)

    #         X = np.array(trimmed_segments)
    #         y = np.array(trimmed_segments_y)
    #         # Podział na dane treningowe i testowe
    #         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #         # Optymalizacja modeli
    #         best_model = None
    #         best_score = 0
    #         best_params = {}

    #     #     for name, model in base_models:
    #     #         print(f"Trenuję model: {name}")

    #     #         # Inicjalizacja GridSearchCV
    #     #         grid_search = GridSearchCV(model, param_grid[name], cv=5,scoring='f1_weighted', n_jobs=4)
    #     #         grid_search.fit(X_train, y_train)

    #     #         # Najlepsze wyniki dla danego modelu
    #     #         print(f"{name} - Best Parameters: {grid_search.best_params_}")
    #     #         print(f"{name} - Best Cross-Validation Score: {grid_search.best_score_}")

    #     #         # Sprawdzanie, czy to najlepszy model
    #     #         if grid_search.best_score_ > best_score:
    #     #             best_model = grid_search.best_estimator_
    #     #             best_score = grid_search.best_score_
    #     #             best_params = grid_search.best_params_

    #     #     # Predykcja i ocena najlepszego modelu
    #     #     print("\nNajlepszy model:", best_model)
    #     #     print("Najlepsze parametry:", best_params)

    #     #     y_pred = best_model.predict(X_test)

    #     #     accuracy = accuracy_score(y_test, y_pred)
    #     #     recall = recall_score(y_test, y_pred, average='weighted')

    #     #     print(f"Accuracy: {accuracy}")
    #     #     print(f"Recall: {recall}")
    
    #     except Exception as e:
    #         print(f"Wystąpił błąd podczas treningu: {e}")




    # def fit(self, X, y):
    #     """
    #     Trenuje model na podanych danych.

    #     Args:
    #         X (np.ndarray): Dane wejściowe (features).
    #         y (np.ndarray): Etykiety (labels).
    #     """
    #     try:
                        
    #         # Definicja modelu
    #         model = OneVsRestClassifier(XGBClassifier(n_jobs=8, max_depth=20, n_estimators=1000,
    #                                                    eval_metric='auc', objective='binary:hinge',
    #                                                    tree_method='hist'))
            
    #         param_grid = {
    #             # 'estimator__max_depth': [5, 10, 15, 20],
    #             'estimator__n_estimators': [500, 1000, 1500],
    #             # 'estimator__learning_rate': [0.01, 0.1, 0.2],
    #         }
                        
    #         # Tworzenie pipeline
    #         one_rep = 50
    #         divison = 1
    #         pipeline = Pipeline([
    #             ('feature_extraction', WindowFeatureExtractor(window_size=one_rep, step_size=one_rep // divison)),
    #             ('label_processing', WindowLabelProcessor(window_size=one_rep, step=one_rep // divison)),
    #             ('classifier', model)
    #         ])
            
    #         # Przekształcanie danych
    #         feature_extractor = pipeline.named_steps['feature_extraction']
    #         label_processor = pipeline.named_steps['label_processing']
    #         classifier = pipeline.named_steps['classifier']
    #         X = feature_extractor.transform(X)
    #         y = label_processor.transform(y)
    #         y = y[:, [0, 1, 2, 3, 5, 7]]
    #         pca = PCA(n_components=50)  # Wybieramy 100 głównych składowych
    #         X = pca.fit_transform(X)

    #         # Podział na dane treningowe i testowe
    #         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
    #         # Inicjalizacja GridSearchCV
    #         grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            
    #         # Dopasowanie modelu
    #         grid_search.fit(X_train, y_train)
            
    #         # Najlepsze parametry
    #         print(f'Best Parameters: {grid_search.best_params_}')
    #         print(f'Best Cross-Validation Score: {grid_search.best_score_}')
            
    #         # Predykcja na zbiorze testowym
    #         y_pred = grid_search.best_estimator_.predict(X_test)
            
    #         # Ocena modelu
    #         accuracy = accuracy_score(y_test, y_pred)
    #         print(f'Accuracy: {accuracy}')
            
    #         recall = recall_score(y_test, y_pred, average='weighted')
    #         print(f'Recall: {recall}')

    #     except Exception as e:
    #         print(f"Błąd podczas trenowania modelu: {e}")

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
            predictions = self.model.predict(X)
            return predictions
        except Exception as e:
            print(f"Błąd podczas przewidywania: {e}")
            return None

    def score(self, X, y):
        """
        Oblicza dokładność modelu na podanych danych testowych.

        Args:
            X (np.ndarray): Dane wejściowe (features).
            y (np.ndarray): Rzeczywiste etykiety (labels).

        Returns:
            float: Dokładność modelu.
        """
        if not self.is_fitted:
            raise ValueError("Model nie został jeszcze wytrenowany. Użyj metody fit przed score.")
        try:
            predictions = self.predict(X)
            accuracy = accuracy_score(y, predictions)
            return accuracy
        except Exception as e:
            print(f"Błąd podczas obliczania dokładności: {e}")
            return None

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

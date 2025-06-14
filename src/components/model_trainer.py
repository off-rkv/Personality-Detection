import os
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import(
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)

from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

from sklearn.preprocessing import LabelEncoder

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):

        try:
            logging.info('split training and test input data')
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(y_train)
            y_test = label_encoder.transform(y_test)

            models=(
                {
                    "Logistic Regression": LogisticRegression(),
                    "K-Nearest Neighbors": KNeighborsClassifier(),
                    "Decision Tree": DecisionTreeClassifier(),
                    "Random Forest": RandomForestClassifier(),
                    "XGBoost": XGBClassifier(),
                    "CatBoost": CatBoostClassifier(verbose=0),
                    "AdaBoost": AdaBoostClassifier(),
                    "Gradient Boosting": GradientBoostingClassifier()
                }
            )
            params={
                "Logistic Regression": {
                    "solver": ["liblinear", "lbfgs"],
                    "C": [0.01, 0.1, 1, 10]
                },
                "K-Nearest Neighbors": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"]
                },
                "Decision Tree": {
                    "criterion": ["gini", "entropy"],
                    "max_depth": [None, 5, 10, 20]
                },
                "Random Forest": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 5, 10, 20]
                },
                "XGBoost": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7]
                },
                "CatBoost": {
                    "iterations": [100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "depth": [3, 5, 7]
                },
                "AdaBoost": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1]
                },
                "Gradient Boosting": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7]
                }
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                              models=models,param=params)
            
            best_model_score=max(model_report.values())

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]
            
            if best_model_score<0.5:
                raise CustomException(f"No suitable model found. Accuracy: {best_model_score}", sys)

            logging.info(f'Best model found: {best_model_name} with Accuracy score: {best_model_score}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            actual_labels = label_encoder.inverse_transform(predicted)
            accuracy = accuracy_score(y_test, predicted)
            save_object("artifacts/label_encoder.pkl", label_encoder)
            return accuracy
        
        except Exception as e:
            raise CustomException(e,sys)
        
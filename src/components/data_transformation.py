#Data transformation(Columns Transformation and Standard Scaler)

import sys
import os
import pandas as pd
import numpy as np

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer


from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import BayesianRidge

from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_transformation_object(self,input_df):
        '''This function is responsible for data transformation'''

        try:
            numerical_columns=input_df.select_dtypes(exclude='O').columns.to_list() # list of all the numerical columns
            categorical_columns=input_df.select_dtypes(include='O').columns.to_list() # list of all the categorical columns

            #Use to pipeline to chain together multiple data transformation 
            logging.info("Pipeline the data transformation steps")

            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler())
                ]
            )
            
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                    ('one_hot_encoder',OneHotEncoder())
                ]
            )
            logging.info("Numerical columns standard scaling completed")

            logging.info("Categorical Columns encoding completed")

            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns),
                    ('cat_pipeline',cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Read train and test data completed")
            logging.info('Obtaining preprocessing object')


            target_column_name='Personality'

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)

            target_feature_train_df = train_df[target_column_name]
            target_feature_test_df = test_df[target_column_name]
            
            preprocessing_obj=self.get_transformation_object(input_feature_train_df)
            
            logging.info(
                f'Applying preprocessor object on training dataframe and testing dataframe'
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            #concatenate
            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]

            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            logging.info(f'Saved preprocessing object & Train shape -> {train_arr.shape} and Test_arr -> {test_arr.shape}.')

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )                        
        
        except Exception as e:
            raise CustomException(e,sys)


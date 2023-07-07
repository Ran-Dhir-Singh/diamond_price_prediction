import sys
import os
import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from dataclasses import dataclass


## Data Transfomation config

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path : str = os.path.join('artifacts','preprocessor.pkl')


## Data Transformation class 

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    
    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')

            # Defining Categorical and Numerical columns
            categorical_columns = ['cut','color','clarity']
            numerical_columns = ['carat','depth','table','x','y','z']

            # Defining the ordinal ranking for ordinal columns
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info('Pipeline Initiated')

            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            # Categorical Pipeline
            cat_pipeline = Pipeline(
                steps= [
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinal_encder', OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ('scaler', StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns) 
            ])

            logging.info('Pipeline Completed')

            return preprocessor
        
        except Exception as e:
            logging.info('Error in Data Transformation')
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        try:

            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading of train and test data completed")
            logging.info(f"Train Dataframe head : \n{train_df.head().to_string()}") 
            logging.info(f"Test Dataframe head : \n{test_df.head().to_string()}")

            logging.info("Obtaining Preprocessing object")

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'price'
            drop_columns = [target_column_name,'id']

            # Depndent and Independent  features
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Applying the transformation

            input_feature_train_array = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_array = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying Preprocessing on train and test dataset")

            train_arr = np.c_[input_feature_train_array, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_array, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessor pickle created and saved")

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        

        except Exception as e:
            logging.info("Exception occured in initiate_data_transformation")
            raise CustomException(e, sys)
        
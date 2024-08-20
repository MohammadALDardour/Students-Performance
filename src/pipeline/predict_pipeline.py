import sys

import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self) -> None:
        pass

    
    def predicted(self, features):
        print(type(features))
        try:
            model = load_object("artifacts\model.pkl")
            preprocessor = load_object("artifacts\preprocessor.pkl")
            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int) -> None:

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score


    def get_data_as_dataFrame(self) -> pd.DataFrame:
        try:
            custom_data = {
                "gender": [self.gender],
                "race/ethnicity": [self.race_ethnicity],  # Updated column name
                "parental level of education": [self.parental_level_of_education],  # Updated column name
                "lunch": [self.lunch],
                "test preparation course": [self.test_preparation_course],  # Updated column name
                "reading score": [self.reading_score],  # Ensure consistency with previous fix
                "writing score": [self.writing_score]   # Ensure consistency with previous fix
            }

            return pd.DataFrame(custom_data)

        except Exception as e:
            raise CustomException(e, sys)
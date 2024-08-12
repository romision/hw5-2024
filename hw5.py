import json
import pathlib
from typing import Union, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class QuestionnaireAnalysis:
    def __init__(self, data_fname: Union[pathlib.Path, str]):
        self.data_fname = pathlib.Path(data_fname)
        if not self.data_fname.exists():
            raise ValueError(f"File {self.data_fname} does not exist.")
        self.data = None

    def read_data(self):
        with open(self.data_fname, 'r') as file:
            self.data = pd.DataFrame(json.load(file))
        
        self.data['age'] = pd.to_numeric(self.data['age'], errors='coerce')
        
        grade_columns = ['q1', 'q2', 'q3', 'q4', 'q5']
        self.data[grade_columns] = self.data[grade_columns].apply(pd.to_numeric, errors='coerce')

    def show_age_distrib(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.data is None:
            raise ValueError("Data not read. Call read_data() before this method.")
        
        ages = self.data['age'].dropna()
        bins = np.arange(0, 101, 10)
        hist, bins = np.histogram(ages, bins=bins)

        plt.hist(ages, bins=bins, edgecolor='black')
        plt.xlabel('Age')
        plt.ylabel('Number of Participants')
        plt.title('Age Distribution of Participants')
        plt.show(block=False)

        return hist, bins

    def remove_rows_without_mail(self) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("Data not read. Call read_data() before this method.")
        
        def is_valid_email(email: str) -> bool:
            if email.count('@') != 1:
                return False
            local, domain = email.split('@')
            if local.startswith('.') or local.endswith('.') or domain.startswith('.') or domain.endswith('.'):
                return False
            if '.' not in domain:
                return False
            return True

        valid_emails = self.data['email'].apply(is_valid_email)
        df = self.data[valid_emails].reset_index(drop=True)

        return df

    def fill_na_with_mean(self) -> Tuple[pd.DataFrame, np.ndarray]:
        if self.data is None:
            raise ValueError("Data not read. Call read_data() before this method.")
        
        df = self.data.copy()
        row_indices = []

        grade_columns = ['q1', 'q2', 'q3', 'q4', 'q5']

        for index, row in df.iterrows():
            if row[grade_columns].isnull().any():
                row_indices.append(index)
                mean_value = row[grade_columns].mean(skipna=True)
                df.loc[index, grade_columns] = row[grade_columns].fillna(mean_value)

        return df, np.array(row_indices)

    def score_subjects(self, maximal_nans_per_sub: int = 1) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("Data not read. Call read_data() before this method.")

        df = self.data.copy()
        grade_columns = ['q1', 'q2', 'q3', 'q4', 'q5']

        def calculate_score(row):
            nan_count = row[grade_columns].isnull().sum()
            if nan_count > maximal_nans_per_sub:
                return pd.NA
            return np.floor(row[grade_columns].mean(skipna=True))

        df['score'] = df.apply(calculate_score, axis=1)
        df['score'] = df['score'].astype('UInt8')
        return df

    def correlate_gender_age(self) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("Data not read. Call read_data() before this method.")
        
        df = self.data.copy()
        df['age'] = df['age'] > 40
        grade_columns = ['q1', 'q2', 'q3', 'q4', 'q5']
        
        # Fill NaN values with mean before grouping
        for col in grade_columns:
            df[col] = df[col].fillna(df[col].mean())
        
        # Group by gender and age, and calculate the mean of the grade columns
        grouped = df.groupby(['gender', 'age'])[grade_columns].mean().round(2)
        
        # Debugging print statements
        print("Grouped DataFrame:")
        print(grouped)
        
        return grouped

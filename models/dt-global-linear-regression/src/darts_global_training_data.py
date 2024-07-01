from enfobench.dataset import Dataset
import pandas as pd
import numpy as np
from darts import TimeSeries
import os
import pickle
from darts.models import RegressionModel
from sklearn.linear_model import BayesianRidge,LinearRegression
from enfobench.evaluation import evaluate_metrics
from enfobench.evaluation.metrics import mean_absolute_error, mean_bias_error,root_mean_squared_error
from enfobench import AuthorInfo, ModelInfo, ForecasterType
from enfobench.evaluation.server import server_factory
from enfobench.evaluation.utils import create_forecast_index, periods_in_duration
from enfobench.dataset.utils import create_perfect_forecasts_from_covariates
from darts.utils.missing_values import missing_values_ratio, fill_missing_values
from enfobench.dataset import Dataset, DemandDataset

unique_ids = ds.metadata_subset.list_unique_ids()

def strip(self) -> "TimeSeries":

    df = self.pd_dataframe(copy=False)
    new_start_idx = df.first_valid_index()
    new_end_idx = df.last_valid_index()
    new_series = df.loc[new_start_idx:new_end_idx]
    return self.__class__.from_dataframe(new_series)

class Darts_Global_Training_Data:

    def __init__(self, unique_ids: list):
        self.unique_ids = unique_ids
        self.target_result = None
        self.strip_series = None

    def target(self):
        
        final_target = []
        
        l = len(unique_ids)

        for i in range(l):
    
            target, past_covariates, metadata = ds.get_data_by_unique_id(self.unique_ids[i])
            final_target.append(target)
            result_df = pd.concat(final_target, axis=1)
            
        self.target_result = result_df 
            
    def strip_time_series(self):
        self.strip_series = [
            strip(TimeSeries.from_series(self.target_result.iloc[:, i], fill_missing_dates=True))
            for i in range(len(self.target_result.columns))
        ]     
    
    def replace_missing_values(self):
        
        final_series = []

        l = len(self.strip_series)
        
        for i in range(l):
            
            time_series = fill_missing_values(self.strip_series[i],method="nearest")
            
            final_series.append(time_series)

        return final_series

    def process_data(self):
        
        self.target()
        self.strip_time_series()
        final_result = self.replace_missing_values()
        return final_result
    

data_instance = Darts_Global_Training_Data(unique_ids)
result_double = data_instance.process_data()


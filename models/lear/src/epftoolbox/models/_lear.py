"""
Classes and functions to implement the LEAR model for electricity price forecasting
"""

# Author: Jesus Lago

# License: AGPL-3.0 License

import numpy as np
import pandas as pd
from statsmodels.robust import mad
import os
import holidays
import joblib

from sklearn.linear_model import LassoLarsIC, Lasso, LassoCV
from epftoolbox.data import scaling
from epftoolbox.data import read_data
from epftoolbox.evaluation import MAE, sMAPE

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


class LEAR(object):
    """Class to build a LEAR model, recalibrate it, and use it to predict DA electricity prices.
    
    An example on how to use this class is provided :ref:`here<learex2>`.
    
    Parameters
    ----------
    calibration_window : int, optional
        Calibration window (in days) for the LEAR model.
        
    """
    
    def __init__(self, calibration_window=364 * 3):

        # Calibration window in hours
        self.calibration_window = calibration_window

    # Ignore convergence warnings from scikit-learn LASSO module

    def recalibrate(self, Xtrain, Ytrain, Feat_selection):

        """Function to recalibrate the LEAR model. 
        
        It uses a training (Xtrain, Ytrain) pair for recalibration
        
        Parameters
        ----------
        Xtrain : numpy.array
            Input in training dataset. It should be of size *[n,m]* where *n* is the number of days
            in the training dataset and *m* the number of input features
        
        Ytrain : numpy.array
            Output in training dataset. It should be of size *[n,24]* where *n* is the number of days 
            in the training dataset and 24 are the 24 prices of each day
                
        Returns
        -------
        numpy.array
            The prediction of day-ahead prices after recalibrating the model        
        
        """

        # # Applying Invariant, aka asinh-median transformation to the prices
        [Ytrain], self.scalerY = scaling([Ytrain], 'Invariant')

        # # Rescaling all inputs except dummies (7 last features)
        [Xtrain_no_dummies], self.scalerX = scaling([Xtrain[:, :-8]], 'Invariant')
        Xtrain[:, :-8] = Xtrain_no_dummies
        self.models = {}
        for h in range(24):

            # Estimating lambda hyperparameter using LARS
            if Feat_selection:
                param_model = LassoLarsIC(criterion='aic', max_iter=5000)
                param = param_model.fit(Xtrain, Ytrain[:, h]).alpha_
                # Re-calibrating LEAR using standard LASSO estimation technique
                model = Lasso(max_iter=5000, alpha=param, tol=0.01)
            else:
                model = LassoCV(cv=5, max_iter=5000) 

            model.fit(Xtrain, Ytrain[:, h])

            self.models[h] = model

    def predict(self, X):
        """Function that makes a prediction using some given inputs.
        
        Parameters
        ----------
        X : numpy.array
            Input of the model.
        
        Returns
        -------
        numpy.array
            An array containing the predictions.
        """

        # Predefining predicted prices
        Yp = np.zeros(24)

        # # Rescaling all inputs except dummies (7 last features)
        X_no_dummies = self.scalerX.transform(X[:, :-8])
        X[:, :-8] = X_no_dummies

        # Predicting the current date using a recalibrated LEAR
        for h in range(24):

            # Predicting test dataset and saving
            Yp[h] = self.models[h].predict(X)
        
        Yp = self.scalerY.inverse_transform(Yp.reshape(1, -1))

        return Yp
    
    def save_model(self, next_day_date):

        date_str = next_day_date.strftime('%Y-%m-%d')
        models_dir = 'saved_models/LEAR'

        model_data = {
            'models': self.models,
            'scalerX': self.scalerX,
            'scalerY': self.scalerY
        }

        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        try:
            joblib.dump(model_data, f'{models_dir}/LEAR_CW{self.calibration_window}_{date_str}.joblib')
        except Exception as e:
            print(f"Failed to save model: {e}")
         
    def recalibrate_predict(self, Xtrain, Ytrain, Xtest, next_day_date, Feat_selection, train):
        """Function that first recalibrates the LEAR model and then makes a prediction.

        The function receives the training dataset, and trains the LEAR model. Then, using
        the inputs of the test dataset, it makes a new prediction.
        
        Parameters
        ----------
        Xtrain : numpy.array
            Input of the training dataset.
        Xtest : numpy.array
            Input of the test dataset.
        Ytrain : numpy.array
            Output of the training dataset.
        
        Returns
        -------
        numpy.array
            An array containing the predictions in the test dataset.
        """
        if train:
            self.recalibrate(Xtrain=Xtrain, Ytrain=Ytrain,Feat_selection=Feat_selection)
            train=False

        Yp = self.predict(X=Xtest)

        self.save_model(next_day_date)

        return Yp, train

    def _build_and_split_XYs(self, df_train, df_test=None, date_test=None):
        
        """Internal function that generates the X,Y arrays for training and testing based on pandas dataframes
        
        Parameters
        ----------
        df_train : pandas.DataFrame
            Pandas dataframe containing the training data
        
        df_test : pandas.DataFrame
            Pandas dataframe containing the test data
        
        date_test : datetime, optional
            If given, then the test dataset is only built for that date
        
        Returns
        -------
        list
            [Xtrain, Ytrain, Xtest] as the list containing the (X,Y) input/output pairs for training, 
            and the input for testing
        """

        # Checking that the first index in the dataframes corresponds with the hour 00:00 
        if df_train.index[0].hour != 0 or df_test.index[0].hour != 0:
            print('Problem with the index')

        # 
        # Defining the number of Exogenous inputs
        n_exogenous_inputs = len(df_train.columns) - 1

        # 96 prices + n_exogenous * (24 * 3 exogeneous) + 7 weekday dummies
        # Price lags: D-1, D-2, D-3, D-7
        # Exogeneous inputs lags: D, D-1, D-7
        n_features = 96 + 8 + n_exogenous_inputs * 72

        # Extracting the predicted dates for testing and training. We leave the first week of data
        # out of the prediction as we the maximum lag can be one week
        
        # We define the potential time indexes that have to be forecasted in training
        # and testing
        indexTrain = df_train.loc[df_train.index[0] + pd.Timedelta(weeks=1):].index

        # For testing, the test dataset is different whether depending on whether a specific test
        # dataset is provided
        if date_test is None:
            indexTest = df_test.loc[df_test.index[0] + pd.Timedelta(weeks=1):].index
        else:
            indexTest = df_test.loc[date_test:date_test + pd.Timedelta(hours=23)].index

        # We extract the prediction dates/days.
        predDatesTrain = indexTrain.round('1h')[::24]                
        predDatesTest = indexTest.round('1h')[::24]

        # We create two dataframe to build XY.
        # These dataframes have as indices the first hour of the day (00:00)
        # and the columns represent the 23 possible horizons/dates along a day
        indexTrain = pd.DataFrame(index=predDatesTrain, columns=['h' + str(hour) for hour in range(24)])
        indexTest = pd.DataFrame(index=predDatesTest, columns=['h' + str(hour) for hour in range(24)])
        for hour in range(24):
            indexTrain.loc[:, 'h' + str(hour)] = indexTrain.index + pd.Timedelta(hours=hour)
            indexTest.loc[:, 'h' + str(hour)] = indexTest.index + pd.Timedelta(hours=hour)

        
        # Preallocating in memory the X and Y arrays          
        Xtrain = np.zeros([indexTrain.shape[0], n_features])
        Xtest = np.zeros([indexTest.shape[0], n_features])
        Ytrain = np.zeros([indexTrain.shape[0], 24])

        # Index that 
        feature_index = 0
        
        #
        # Adding the historial prices during days D-1, D-2, D-3, and D-7
        #

        # For each hour of a day
        for hour in range(24):
            # For each possible past day where prices can be included
            for past_day in [1, 2, 3, 7]:

                # We define the corresponding past time indexs using the auxiliary dataframses 
                pastIndexTrain = pd.to_datetime(indexTrain.loc[:, 'h' + str(hour)].values) - \
                    pd.Timedelta(hours=24 * past_day)
                pastIndexTest = pd.to_datetime(indexTest.loc[:, 'h' + str(hour)].values) - \
                    pd.Timedelta(hours=24 * past_day)

                # We include the historical prices at day D-past_day and hour "h" 
                Xtrain[:, feature_index] = df_train.loc[pastIndexTrain, 'Target']
                Xtest[:, feature_index] = df_test.loc[pastIndexTest, 'Target']
                feature_index += 1

        #
        # Adding the exogenous inputs during days D, D-1,  D-7
        #
        # For each hour of a day
        for hour in range(24):
            # For each possible past day where exogenous inputs can be included
            for past_day in [1, 7]:
                # For each of the exogenous input
                for exog in range(1, n_exogenous_inputs + 1):

                    # Definying the corresponding past time indexs using the auxiliary dataframses 
                    pastIndexTrain = pd.to_datetime(indexTrain.loc[:, 'h' + str(hour)].values) - \
                        pd.Timedelta(hours=24 * past_day)
                    pastIndexTest = pd.to_datetime(indexTest.loc[:, 'h' + str(hour)].values) - \
                        pd.Timedelta(hours=24 * past_day)

                    # Including the exogenous input at day D-past_day and hour "h" 
                    Xtrain[:, feature_index] = df_train.loc[pastIndexTrain, 'Exogenous ' + str(exog)]                    
                    Xtest[:, feature_index] = df_test.loc[pastIndexTest, 'Exogenous ' + str(exog)]
                    feature_index += 1

            # For each of the exogenous inputs we include feature if feature selection indicates it
            for exog in range(1, n_exogenous_inputs + 1):
                
                # Definying the corresponding future time indexs using the auxiliary dataframses 
                futureIndexTrain = pd.to_datetime(indexTrain.loc[:, 'h' + str(hour)].values)
                futureIndexTest = pd.to_datetime(indexTest.loc[:, 'h' + str(hour)].values)

                # Including the exogenous input at day D and hour "h" 
                Xtrain[:, feature_index] = df_train.loc[futureIndexTrain, 'Exogenous ' + str(exog)]        
                Xtest[:, feature_index] = df_test.loc[futureIndexTest, 'Exogenous ' + str(exog)] 
                feature_index += 1

        #
        # Adding the dummy variables that depend on the day of the week. Monday is 0 and Sunday is 6
        #
        # For each day of the week
        for dayofweek in range(7):
            Xtrain[indexTrain.index.dayofweek == dayofweek, feature_index] = 1
            Xtest[indexTest.index.dayofweek == dayofweek, feature_index] = 1
            feature_index += 1

        # Extracting the predicted values Y
        for hour in range(24):
            # Defining time index at hour h
            futureIndexTrain = pd.to_datetime(indexTrain.loc[:, 'h' + str(hour)].values)
            futureIndexTest = pd.to_datetime(indexTest.loc[:, 'h' + str(hour)].values)

            # Extracting Y value based on time indexs
            Ytrain[:, hour] = df_train.loc[futureIndexTrain, 'Target']        
    
        belgium_holidays = holidays.Belgium()

        def is_holiday(date):
            return date in belgium_holidays

        # Adding the holiday dummy variable for Xtrain
        for i, date in enumerate(indexTrain.index.date):
            Xtrain[i, -1] = is_holiday(date)

        # Adding the holiday dummy variable for Xtest
        for i, date in enumerate(indexTest.index.date):
            Xtest[i, -1] = is_holiday(date)

        return Xtrain, Ytrain, Xtest


    def select_features(self, df):
        important_features = ['y','soil_temperature_7_to_28cm', 'dew_point_2m', 'soil_temperature_0_to_7cm', 'soil_moisture_7_to_28cm', 'apparent_temperature']

        filtered_df = df[important_features]

        return filtered_df

    def predict_with_horizon(self, df, hourly_forecast_index, forecast_horizon_steps, Feat_selection, train):

        df=self.select_features(df)
        n_exogeneous_inputs = len(df.columns) - 1
        columns = ['Target'] + ['Exogenous ' + str(n) for n in range(1, n_exogeneous_inputs + 1)]
        df.columns = columns
        initial_date=pd.Timestamp(hourly_forecast_index[0])
        next_day_date = pd.Timestamp(initial_date.date())
        predictions = []
        i=0
        while forecast_horizon_steps > 0:

            df_train = df.loc[:next_day_date - pd.Timedelta(hours=1)]
            df_train = df_train.iloc[-self.calibration_window * 24:]
            df_test = df.loc[next_day_date - pd.Timedelta(weeks=2):next_day_date + pd.Timedelta(hours=23), :]

            Xtrain, Ytrain, Xtest = self._build_and_split_XYs(df_train, df_test, next_day_date)
                 
            Yp, train = self.recalibrate_predict(Xtrain=Xtrain, Ytrain=Ytrain, Xtest=Xtest, next_day_date=next_day_date, Feat_selection=Feat_selection, train=train)

            if i==0:
                start_hour = initial_date.hour
                predicted_hours = 24 - start_hour
                Forecast=Yp[0][start_hour:start_hour + predicted_hours]
                date_forecast=initial_date

            elif forecast_horizon_steps<24:
                Forecast=Yp[0][:forecast_horizon_steps]
                predicted_hours=forecast_horizon_steps
                date_forecast=next_day_date
            else:
                Forecast=Yp[0]
                predicted_hours=24
                date_forecast=next_day_date

            predictions.extend(Forecast)
            prediction_dates = pd.date_range(date_forecast, periods=predicted_hours, freq='1h')
            df.loc[prediction_dates, 'Target'] = Forecast
            # Update the next_day_date and reduce the forecast horizon
            next_day_date += pd.Timedelta(hours=24)
            forecast_horizon_steps -= predicted_hours
            i=1

        return predictions

def evaluate_lear_in_test_dataset(path_datasets_folder=os.path.join('.', 'datasets'), 
                                  path_recalibration_folder=os.path.join('.', 'experimental_files'),
                                  dataset='PJM', years_test=2, calibration_window=364 * 3, 
                                  begin_test_date=None, end_test_date=None):
    """Function for easy evaluation of the LEAR model in a test dataset using daily recalibration. 
    
    The test dataset is defined by a market name and the test dates dates. The function
    generates the test and training datasets, and evaluates a LEAR model considering daily recalibration. 
    
    An example on how to use this function is provided :ref:`here<learex1>`.   

    Parameters
    ----------
    path_datasets_folder : str, optional
        path where the datasets are stored or, if they do not exist yet,
        the path where the datasets are to be stored.
    
    path_recalibration_folder : str, optional
        path to save the files of the experiment dataset.
    
    dataset : str, optional
        Name of the dataset/market under study. If it is one one of the standard markets, 
        i.e. ``"PJM"``, ``"NP"``, ``"BE"``, ``"FR"``, or ``"DE"``, the dataset is automatically downloaded. If the name
        is different, a dataset with a csv format should be place in the ``path_datasets_folder``.

    years_test : int, optional
        Number of years (a year is 364 days) in the test dataset. It is only used if 
        the arguments ``begin_test_date`` and ``end_test_date`` are not provided.
    
    calibration_window : int, optional
        Number of days used in the training dataset for recalibration.
    
    begin_test_date : datetime/str, optional
        Optional parameter to select the test dataset. Used in combination with the argument
        ``end_test_date``. If either of them is not provided, the test dataset is built using the 
        ``years_test`` argument. ``begin_test_date`` should either be a string with the following 
        format ``"%d/%m/%Y %H:%M"``, or a datetime object.
    
    end_test_date : datetime/str, optional
        Optional parameter to select the test dataset. Used in combination with the argument
        ``begin_test_date``. If either of them is not provided, the test dataset is built using the 
        ``years_test`` argument. ``end_test_date`` should either be a string with the following 
        format ``"%d/%m/%Y %H:%M"``, or a datetime object.       
    
    Returns
    -------
    pandas.DataFrame
        A dataframe with all the predictions in the test dataset. The dataframe is also written to path_recalibration_folder.
    """

    # Checking if provided directory for recalibration exists and if not create it
    if not os.path.exists(path_recalibration_folder):
        os.makedirs(path_recalibration_folder)

    # Defining train and testing data
    df_train, df_test = read_data(dataset=dataset, years_test=years_test, path=path_datasets_folder,
                                  begin_test_date=begin_test_date, end_test_date=end_test_date)

    # Defining unique name to save the forecast
    forecast_file_name = 'LEAR_forecast' + '_dat' + str(dataset) + '_YT' + str(years_test) + \
                         '_CW' + str(calibration_window) + '.csv'

    forecast_file_path = os.path.join(path_recalibration_folder, forecast_file_name)


    # Defining empty forecast array and the real values to be predicted in a more friendly format
    forecast = pd.DataFrame(index=df_test.index[::24], columns=['h' + str(k) for k in range(24)])
    real_values = df_test.loc[:, ['Target']].values.reshape(-1, 24)
    real_values = pd.DataFrame(real_values, index=forecast.index, columns=forecast.columns)

    forecast_dates = forecast.index

    model = LEAR(calibration_window=calibration_window)

    # For loop over the recalibration dates
    for date in forecast_dates:

        # For simulation purposes, we assume that the available data is
        # the data up to current date where the prices of current date are not known
        data_available = pd.concat([df_train, df_test.loc[:date + pd.Timedelta(hours=23), :]], axis=0)

        # We set the real prices for current date to NaN in the dataframe of available data
        data_available.loc[date:date + pd.Timedelta(hours=23), 'Target'] = np.NaN

        # Recalibrating the model with the most up-to-date available data and making a prediction
        # for the next day
        Yp = model.recalibrate_and_forecast_next_day(df=data_available, next_day_date=date, 
                                                     calibration_window=calibration_window)
        # Saving the current prediction
        forecast.loc[date, :] = Yp

        # Computing metrics up-to-current-date
        mae = np.mean(MAE(forecast.loc[:date].values.squeeze(), real_values.loc[:date].values)) 
        smape = np.mean(sMAPE(forecast.loc[:date].values.squeeze(), real_values.loc[:date].values)) * 100

        # Pringint information
        print('{} - sMAPE: {:.2f}%  |  MAE: {:.3f}'.format(str(date)[:10], smape, mae))

        # Saving forecast
        forecast.to_csv(forecast_file_path)

    return forecast

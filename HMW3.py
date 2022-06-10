import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import flow, task
from logging import getLogger
from prefect.task_runners import SequentialTaskRunner

from datetime import date
import datetime
import dateutil.relativedelta
import pickle
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.task_runners import SequentialTaskRunner
from prefect.flow_runners import SubprocessFlowRunner

logger = getLogger("my-logger")
logger.setLevel("INFO")
@task
def read_data(path):
    df = pd.read_parquet(path)
    return df
@task
def prepare_features(df, categorical, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df
@task
def train_model(df, categorical):

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv
@task()
def run_model(df, categorical, dv, lr):
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return

def get_paths(date):
    if date is None:
        current_date = datetime.datetime.strptime(date.today(), "%Y-%m-%d")
    else:
        current_date = datetime.datetime.strptime(date, "%Y-%m-%d")
    train_date = current_date - dateutil.relativedelta.relativedelta(months=2)
    val_date = current_date - dateutil.relativedelta.relativedelta(months=1)
    date_format = '%Y-%m'
    train_date = train_date.strftime(date_format)
    val_date = val_date.strftime(date_format)
    train_path: str = './data/fhv_tripdata_'+train_date+'.parquet'
    val_path: str = './data/fhv_tripdata_'+val_date+'.parquet'
    logger.info(f"Train path is : {train_path}")
    logger.info(f"Val path is : {val_path}")


    return train_path, val_path

@flow(task_runner=SequentialTaskRunner())
def main(date="2021-03-15"):
   # train_path: str = './data/fhv_tripdata_2021-01.parquet', 
    #       val_path: str = './data/fhv_tripdata_2021-02.parquet'):
    train_path, val_path = get_paths(date)
    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    with open('./model/dv-'+ date +'.dv', 'wb') as f_out:
        pickle.dump(dv, f_out)
    with open('./model/model-'+ date +'.bin', 'wb') as f_out:
        pickle.dump(lr, f_out)
    run_model(df_val_processed, categorical, dv, lr)


DeploymentSpec(
    name="cron-schedule-deployment",
    flow = main,
    schedule=CronSchedule(
        cron="0 9 15 * *",
        timezone="America/New_York"),
    flow_runner=SubprocessFlowRunner(),
    tags = ["mlops"]
)


from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

from src.lab import (
    generate_data,
    load_data,
    data_preprocessing,
    build_save_model,
    load_model_summary,
)

default_args = {
    'owner': 'student',
    'start_date': datetime(2024, 1, 1),
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'Airflow_Lab1_Stock_Clustering',
    default_args=default_args,
    description='Synthetic OHLCV stock clustering with Silhouette Score selection',
    schedule_interval=None,
    catchup=False,
)

generate_data_task = PythonOperator(
    task_id='generate_data_task',
    python_callable=generate_data,
    dag=dag,
)

load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data,
    dag=dag,
)

data_preprocessing_task = PythonOperator(
    task_id='data_preprocessing_task',
    python_callable=data_preprocessing,
    op_args=[load_data_task.output],
    dag=dag,
)

build_save_model_task = PythonOperator(
    task_id='build_save_model_task',
    python_callable=build_save_model,
    op_args=[data_preprocessing_task.output, "stock_model.sav"],
    provide_context=True,
    dag=dag,
)

load_model_task = PythonOperator(
    task_id='load_model_task',
    python_callable=load_model_summary,
    op_args=["stock_model.sav", build_save_model_task.output],
    dag=dag,
)

generate_data_task >> load_data_task >> data_preprocessing_task >> build_save_model_task >> load_model_task

if __name__ == "__main__":
    dag.cli()
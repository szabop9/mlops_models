from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="pull_mlops_dev_branch",
    default_args=default_args,
    description="Pull latest changes from dev branch of mlops_models repo",
    schedule_interval="@daily",  # or use None for manual triggering
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["git", "mlops"],
) as dag:

    pull_repo_task = BashOperator(
        task_id="pull_latest_from_dev",
        bash_command="bash -c ../../pull_dev_branch.sh",
    )

    pull_repo_task
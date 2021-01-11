#---------------------------------------------------
# file: nx-server.py
# desc: performs the nx functions for client apps
#---------------------------------------------------
import logging
import json
import requests
import time
import networkx as nx
import nx_functions
from airflow.models import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import BranchPythonOperator
from requests.exceptions import HTTPError


#------------------------------------------------------------------
# name: nx_trigger_new_service_handler()
# desc: triggers nx server to run again after a some elapse time
#------------------------------------------------------------------
def nx_trigger_new_service_handler(ds, **kwargs): 
    timelimit = 1
    start_time = time.time()
    while((time.time() - start_time) <= timelimit):
        elasped_time = time.time() - start_time
        print("Waiting...................")
        btimeout = elasped_time <= timelimit
    # end while 

    # trigger this dag to run again
    request_url = "http://localhost:8080/api/experimental/dags/nx-server/dag_runs"
    nx_functions.nx_do_request(request_url, {"none":"none"})
    print("Trigger nx-server dag to run again.")
# end nx_trigger_new_service_handler()


#---------------------------------------------------------
# name: nx_service_handler()
# desc: carries out the service. ex.) erdos-renyi model
#---------------------------------------------------------
def nx_service_handler(ds, **kwargs):	
    response = nx_functions.nx_set_call_output(nx_functions.nx_exec_call(nx_functions.nx_get_call_input()))
    print("nx_set_call_output(nx_exec_call(nx_get_call_input())): ", response)
    return "nx-service-response" if(response) else "nx-service-no-response"


###############################################################
## defines the dag that is scheduled to do networkx services
dag = DAG(
    dag_id='nx-server',
    default_args={
        'owner': 'Airflow',
        'start_date': days_ago(2),
    },
    schedule_interval=None
) ## end DAG


###############################################################
## set up the dags task
nx_service = BranchPythonOperator(
    task_id='nx-service',
    provide_context=True,
    python_callable=nx_service_handler,
    dag=dag
) ## run the service in parallel

nx_service_handler_response = DummyOperator(
    task_id='nx-service-response',
    dag=dag
) ## run if nx_service has a response

nx_service_handler_no_response = DummyOperator(
    task_id='nx-service-no-response',
    dag=dag
) ## run if nx_service has no response

nx_trigger_new_service = PythonOperator(
    task_id='nx-trigger-new-service',
    provide_context=True,
    python_callable=nx_trigger_new_service_handler,
    dag=dag
) ## run the service in parallel

# set up DAG order
nx_service >> nx_service_handler_response
nx_service >> nx_service_handler_no_response
nx_trigger_new_service 

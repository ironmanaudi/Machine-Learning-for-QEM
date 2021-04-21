import qiskit
from qiskit import *

resdata = []
provider = IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
backend = provider.get_backend('ibmq_16_melbourne')

n = 0
for ran_job in reversed(backend.jobs(limit=3)):
    #print(n, str(ran_job.job_id()) + " " + str(ran_job.status()))
    print(n,'aaaaaaaaaaaaaaaaaa', ran_job.result())
    resdata.append([(ran_job.result())])
    #print(ran_job.result())
    n = n + 1
#print(resdata)


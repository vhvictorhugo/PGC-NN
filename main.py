import sys
import ast
import os
import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from job.poi_categorization_job import PoiCategorizationJob
from foundation.configuration.input import Input

def start_input(args):
    Input().set_inputs(args)


def start_job(args):

    start_input(args)
    job_name = Input.get_instance().inputs['job']
    print(job_name)
    if job_name == "categorization":
        job = PoiCategorizationJob()

    job.start()

if __name__ == "__main__":
    try:

        args = ast.literal_eval(sys.argv[1])
        start_job(args)

    except Exception as e:
        raise e
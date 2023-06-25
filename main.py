import os
import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from job.poi_categorization_job import PoiCategorizationJob
from job.matrix_generation_for_poi_categorization_job import MatrixGenerationForPoiCategorizationJob


if __name__ == "__main__":
    job = PoiCategorizationJob()
    # job = MatrixGenerationForPoiCategorizationJob()
    
    job.start()
import os
import json
from typing import Dict
from pathlib import Path
from loguru import logger
from src.verification import ModelStateCheck
from fastapi.security.api_key import APIKeyHeader
from fastapi import FastAPI, HTTPException, Depends, Request

# initialize the fast API app
app = FastAPI()

# set the API KEY
API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="x-key", auto_error=False)

# directories that are mounted to the file system
DATA_DIR = Path("data/")
OUTPUT_DIR = Path("output/")

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Could not validate credentials")
    return api_key

@app.get("/")
def read_root():
    return {"message": "TRAINING API IS FUNCTIONAL"}

@app.get("/health")
def health_check():
    return {"status": "Healthy"}

@app.post("/train", dependencies=[Depends(verify_api_key)])
async def train(request: Request):
    '''
    function that runs the training procedure
    '''
    try:
        
        # await the payload
        payload = await request.json()

        # get the job id
        job_id = payload.get('job_id')

        # define the pth
        job_pth = OUTPUT_DIR / "training" / job_id
        local_pth = job_pth / "local_training.json"
        model_pth = job_pth / "model.json"

        # initialize the checker 
        model_checker = ModelStateCheck(local_pth, model_pth)

        # run the verification
        model_checker.call(payload)

        # load the model

        # train the model

        logger.info(f"Trained for job_id: {job_id}")
        
        {"message": f"Trained for job_id: {job_id}"}


    except Exception as e:
        logger.error(f"Failed to train when instructed")
        raise HTTPException(
            status_code=500, 
            detail=f"""
                Failed to train when instructed
            """
        )


@app.post("/configure", dependencies=[Depends(verify_api_key)])
async def configure(request: Request):
    '''
    function that configures the FL training procedure
    '''
    try:
        
        # await the payload
        payload = await request.json()
        
        # dowload the configuration information
        local = payload.get('local')
        federated = payload.get('federated')
        state_dict = payload.get('model')

        # retrive the job_id and the model_id
        job_id = federated["job_id"]

        # create the directory for the training job if it does not exist
        new_folder_pth = OUTPUT_DIR / "training" / job_id
        new_folder_pth.mkdir(parents=True, exist_ok=True)

        # infromation to save
        configuration = {
            new_folder_pth / "local_training.json": local,
            new_folder_pth / "federated_training.json": federated,
            new_folder_pth / "model.json": state_dict
        }

        # save the configuration
        for pth, data in configuration.items():
            with open(pth, "w") as json_file:
                json.dump(data, json_file, indent=4)
                json_file.close()
                logger.info(f"SAVED FILE: {pth}")

        logger.info(f"Successfully registered the training procedure for job_id: {job_id}")

        return {"message": f"Successfully registered the training procedure for job_id: {job_id}"}

    except Exception as e:
        logger.error(f"Failed to register the training procedure")
        raise HTTPException(
            status_code=500, 
            detail=f"""
                Failed to register the training procedure
            """
        )


    



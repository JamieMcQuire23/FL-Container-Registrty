import os
import json
from typing import Dict
from pathlib import Path
from loguru import logger
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
        federated= payload.get('federated')

        # retrive the job_id and the model_id
        job_id = federated["job_id"]

        # create the directory for the training job if it does not exist
        new_folder_path = OUTPUT_DIR / "training" / job_id
        new_folder_path.mkdir(parents=True, exist_ok=True)

        # infromation to save
        configuration = {
            new_folder_path / "local_training.json": local,
            new_folder_path / "federated_training.json": federated
        }

        # save the configuration
        for pth, data in configuration.items():
            with open(pth, "w") as json_file:
                json.dump(data, json_file, indent=4)
                json_file.close()
                logger.info(f"SAVED FILE: {pth}")

        return {"message": "successfully registered the configuration"}

    except Exception as e:
        logger.error("FAILED TO REGISTER FL TRAINING PROCEDURE")
        raise HTTPException(
            status_code=500, 
            detail=f"""
                failed to register the configuration... \\
                please confirm you have the local and federated \\
                fields with the job_id specified in the federated field
            """
        )


    



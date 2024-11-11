'''
functions that run the on-device training
'''

import json
import torch
from loguru import logger

class ModelStateCheck:
    '''
    A class that checks whether the incoming payload matches the expected input/output 
    dimensions and types, and also checks model consistency.
    '''

    def __init__(self, local_pth, model_pth):
        """
        Initializes the ModelStateCheck with local and model configurations.

        :param local_pth: JSON string with local configuration (input/output dims, types)
        :param model_pth: JSON string with model configuration (parameters)
        """

        try:
            self.local_config = json.loads(local_pth)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding local_pth JSON: {str(e)}")
            raise

        try:
            self.state_dict = json.loads(model_pth)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding model_pth JSON: {str(e)}")
            raise

        # check the model state is a dict
        if not isinstance(self.state_dict, dict):
            logger.error("Model parameters (state_dict) should be a dictionary.")
            raise ValueError("Model parameters (state_dict) should be a dictionary.")
        
    def call(self, payload):

        self.__check_training_information(self, payload)
        logger.info("Checked training information")
        self.__check_model_consistency(self, payload)
        logger.info("Checked model consistency")

    def __check_training_information(self, payload):

        if 'local' not in payload:
            logger.error("Missing local information")
            raise KeyError(f"Missing local information in payload")
        
        for key, item in payload:
            if key not in self.local_config:
                logger.error(f"Missing parameter: {key}")
                raise KeyError(f"Missing parameter: {key}")
            
            if item != payload[key]:
                logger.error(f"Value mismatch for {key}: expected {payload[key]}, got {item}")
                raise ValueError(f"Value mismatch for {key}: expected {payload[key]}, got {item}")
            
            logger.info(f"Local information {key} is consistent.")

    def __check_model_consistency(self, payload):
        """
        This method checks if the state_dict from the payload matches the initialized model's state_dict.

        :param incoming_model: The model parameters provided in the payload under 'model'.
        """
        
        logger.info("Checking model consistency...")

        if 'model' not in payload:
            logger.error("Missing model payload")
            raise KeyError(f"Missing model state dict in payload")

        state_dict = payload['model']

        # check for the parameters and see if they line up
        for key, value in self.state_dict.items():
            if key not in state_dict:
                logger.error(f"Missing parameter: {key}")
                raise KeyError(f"Missing parameter: {key}")

            # get the tensor from the state dict
            incoming_tensor = torch.tensor(state_dict[key])

            # test for type
            if value.type != incoming_tensor.type:
                logger.error(f"Type mismatch for {key}: expected {value.type}, got {incoming_tensor.type}")
                raise TypeError(f"Shape mismatch for {key}: expected {value.shape}, got {incoming_tensor.shape}")

            # compare parameter shapes
            if value.shape != incoming_tensor.shape:
                logger.error(f"Shape mismatch for {key}: expected {value.shape}, got {incoming_tensor.shape}")
                raise ValueError(f"Shape mismatch for {key}: expected {value.shape}, got {incoming_tensor.shape}")

            logger.info(f"Model parameter {key} is consistent.")

        # Check if there are any extra parameters in the incoming model
        for key in state_dict:
            if key not in self.state_dict:
                logger.error(f"Unexpected parameter in incoming model: {key}")
                raise ValueError(f"Unexpected parameter in incoming model: {key}")

        logger.info("Model consistency check completed.")



    

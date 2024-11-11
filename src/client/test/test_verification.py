import json
import pytest
from src.verification import ModelStateCheck

'''
class TestModelStateChecker:

    def test_model_state_check_initialization():
        """ 
        Test the initialization of the ModelStateCheck class to ensure it correctly parses 
        the local_pth and model_pth JSON strings, loads them into local_config and state_dict, 
        and verifies the types are expected.
        """

        pass


    def test_missing_model_key():
        """
        Test that a KeyError is raised when the 'model' key is missing in the payload.
        """
        pass


    def test_missing_local_key():
        """
        Test that a KeyError is raised when the 'local' key is missing in the payload.
        """
        pass


    def test_type_mismatch_in_model():
        """
        Test that a TypeError is raised when the model parameter type in the payload 
        does not match the expected tensor type.
        """
        pass


    def test_shape_mismatch_in_model():
        """
        Test that a ValueError is raised when the shape of a model parameter in the payload 
        does not match the shape of the corresponding parameter in the state_dict.
        """
        pass


    def test_model_consistency_check_success():
        """
        Test that the model consistency check successfully passes when the payload is consistent, 
        and verify the logger outputs 'Model consistency check completed.'.
        """
        pass


    def test_extra_parameter_in_model():
        """
        Test that a ValueError is raised when an unexpected parameter is found in the incoming model 
        that is not present in the expected state_dict.
        """
        pass


    def test_missing_parameter_in_model():
        """
        Test that a KeyError is raised when a parameter is missing from the model in the payload.
        """
        pass


    def test_payload_with_valid_data():
        """
        Test that the ModelStateCheck class passes when the payload has all the required keys 
        (local, federated, and model) with valid data matching the expected types and shapes.
        """
        pass


    def test_invalid_json_format_in_local_pth():
        """
        Test that a JSONDecodeError is raised when the local_pth JSON is improperly formatted.
        """
        pass


    def test_invalid_json_format_in_model_pth():
        """
        Test that a JSONDecodeError is raised when the model_pth JSON is improperly formatted.
        """
        pass


    def test_check_training_information_missing_local_key():
        """
        Test that a KeyError is raised if a key in the local configuration is missing or 
        does not match the expected value from local_pth.
        """
        pass
'''
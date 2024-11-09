import pytest
from src.api import app
from fastapi import FastAPI
from fastapi.testclient import TestClient

client = TestClient(app)

class TestEndPoints:

    def test_health(self):
        response = client.get("/health")
        assert response.json() == {"status": "Healthy"}

    def test_root(self):
        response = client.get("/")
        assert response.json() == {"message": "TRAINING API IS FUNCTIONAL"}


class TestConfigurationMethods:

    def test_configure_pass(self):

        payload = {
            "local": {
                "lr": 1e-3,
                "batch_size": 512,
                "patience": 10
            },
            "federated": {
                "job_id": "test",
                "rounds": 100,
                "output": ["weights"]
            }
        }

        # post the response
        response = client.post(
            "/configure",
            headers={'x-key': '6b8e92be-fc0a-4db5-b8dc-c35dc05d1108'},
            json=payload
        )

        assert response.status_code == 200
        assert response.json() == {"message": "successfully registered the configuration"}

    def test_configure_failure(self):

        payload = {
            "local": {
                "lr": 1e-3,
                "batch_size": 512,
                "patience": 10
            }
        }

        # post the response
        response = client.post(
            "/configure",
            headers={'x-key': '6b8e92be-fc0a-4db5-b8dc-c35dc05d1108'},
            json=payload
        )

        assert response.status_code == 500
        assert response.json() == {
            "detail": f"""
                failed to register the configuration... \\
                please confirm you have the local and federated \\
                fields with the job_id specified in the federated field
            """
        }

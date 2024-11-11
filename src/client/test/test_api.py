import os
import json
import pytest
from src.api import app
from fastapi.testclient import TestClient

API_KEY = os.getenv("API_KEY")
client = TestClient(app)


class TestEndPoints:

    def test_health(self):
        response = client.get("/health")
        assert response.json() == {"status": "Healthy"}

    def test_root(self):
        response = client.get("/")
        assert response.json() == {"message": "TRAINING API IS FUNCTIONAL"}


class TestConfigurationMethods:

    def test_configure_pass(self, create_payload):

        # create payload for configuration
        payload = create_payload()

        job_id = payload['federated']['job_id']

        # post the response
        response = client.post(
            "/configure",
            headers={'x-key': API_KEY},
            json=payload
        )

        assert response.status_code == 200
        assert response.json() == {"message": f"Successfully registered the training procedure for job_id: {job_id}"}

    def test_configure_failure(self, create_payload):

        # drop the federated information from payload
        payload = create_payload(drop=["federated"])

        # post the response
        response = client.post(
            "/configure",
            headers={'x-key': API_KEY},
            json=payload
        )

        assert response.status_code == 500
        assert response.json() == {
            "detail": f"""
                Failed to register the training procedure
            """
        }

    def test_key_issue(self, create_payload):

        payload = create_payload()

        # post the response
        response = client.post(
            "/configure",
            json=payload
        )

        # post the response
        response = client.post(
            "/configure",
            headers={'x-key': 'silly-me-i-put-the-wrong-key'},
            json=payload
        )

        assert response.status_code == 403

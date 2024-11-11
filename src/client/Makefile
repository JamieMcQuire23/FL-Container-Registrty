APP_NAME=flopper-client-api
VERSION=0.0.0
API_KEY=6b8e92be-fc0a-4db5-b8dc-c35dc05d1108
MODEL_PTH=/app/output/models 

all:
	make dev-build dev-run dev-test

dev-build:
	docker build . -t "${APP_NAME}_v${VERSION}"

dev-run:
	docker run --name "${APP_NAME}" -e API_KEY="${API_KEY}" -d "${APP_NAME}_v${VERSION}"

dev-test:
	@echo "Getting container ID for ${APP_NAME}_v${VERSION}"
	@CONTAINER_ID=$(shell docker ps -q -f "name=${APP_NAME}") && \
	if [ -z "$$CONTAINER_ID" ]; then echo "Error: Container not found"; exit 1; fi && \
	echo "Running tests in container $$CONTAINER_ID" && \
	docker exec -it $$CONTAINER_ID pytest -v

stop:
	@echo "Getting container ID for ${APP_NAME}_v${VERSION}"
	@CONTAINER_ID=$(shell docker ps -q -f "name=${APP_NAME}") && \
	docker stop "${APP_NAME}" && \
	echo "Removing container $$CONTAINER_ID" && \
	docker rm -f "${APP_NAME}"

clean:
	docker rm -f $(docker ps -a -q)

LOCAL_TAG:=$(shell date +"%Y-%m-%d-%H-%M")
LOCAL_IMAGE_NAME:=xgoals_prediction_model:${LOCAL_TAG}

test:
	pytest tests/

format-check:
	isort model
	black model
	pylint --recursive=y model

train:
	format-check test
	bash src/pipeline/train.sh

build: train
	docker build -t ${LOCAL_IMAGE_NAME} .

integration_test: build
	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} bash integration_test/run.sh

publish: build integration_test
	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} bash scripts/publish.sh

setup:
	pipenv install --dev
	pre-commit install


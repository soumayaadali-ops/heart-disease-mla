.PHONY: setup train artifacts registry drift serve pipeline

setup:
	pip install -r requirements.txt
	pip install evidently scipy

train:
	python train.py

artifacts:
	python mlops_artifacts.py

registry:
	python mlops_registry.py

drift:
	python mlops_drift.py

serve:
	mlflow models serve -m "models:/HeartDisease_Production/Production" --port 1235 --no-conda

pipeline: train artifacts registry drift
	@echo "Pipeline MLOps complet execute avec succes !"
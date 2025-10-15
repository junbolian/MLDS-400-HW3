PY=python
VENV=.venv
PIP=$(VENV)/Scripts/pip.exe

venv:
	$(PY) -m venv $(VENV)

install: venv
	$(PIP) install -r requirements.txt

run:
	$(VENV)/Scripts/python.exe src/train.py

docker-build:
	docker build -t mlds-hw3:latest .

docker-run:
	docker run --rm mlds-hw3:latest


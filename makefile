install:
		pip install --upgrade pip &&\
			pip install -r llm_app/requirements.txt

format:
		black llm_app/*.py

lint:
		pylint --disable=R,C llm_app/*.py  || true



all: install format lint 
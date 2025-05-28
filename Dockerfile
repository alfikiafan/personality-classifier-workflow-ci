FROM continuumio/miniconda3

WORKDIR /app

COPY MLProject/ /app/

RUN conda env create -f conda.yaml

SHELL ["conda", "run", "-n", "workflow-ci-env", "/bin/bash", "-c"]

CMD ["python", "modelling.py"]

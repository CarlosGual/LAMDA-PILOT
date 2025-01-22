FROM nvcr.io/nvidia/pytorch:24.12-py3
LABEL authors="Carlos Gutiérrez"

# Add user
RUN useradd -u 1004 --create-home carlos # 1001 for penelope, 1001 for zeus and 1004 for DGX
WORKDIR /home/carlos
RUN mkdir code/

# Change user
USER carlos

# Install wandb and other aditional libraries
COPY requirements.txt /home/carlos
RUN /bin/bash -c "pip install -r requirements.txt"

COPY entrypoint.sh /home/carlos
ENTRYPOINT [ "/home/carlos/entrypoint.sh" ]
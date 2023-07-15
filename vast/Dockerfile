FROM pytorch/pytorch:latest

# Create conda env and set as default in bash
RUN conda create --name ave python==3.11
RUN conda init bash
RUN echo "conda activate ave" >> ~/.bashrc # FIXME: Doesn't work on vastai


# Install packages
RUN conda install -c conda-forge -n ave git

# Install project dependencies
WORKDIR /root/activation_additions
COPY . .
RUN conda run -n ave pip install --no-cache-dir -e '.[dev]'

# Entrypoint to interactive shell
ENTRYPOINT ["/bin/bash", "-li"]

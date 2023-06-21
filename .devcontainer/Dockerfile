FROM pytorch/pytorch:latest

# Set up non-root user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    apt-get update && \
    apt-get install -y sudo && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME


# Install packages
RUN apt-get install -y git

# Set up the user and environment
USER $USERNAME
ENV PATH="/home/$USERNAME/.local/bin:${PATH}"

# Install project dependencies
WORKDIR /home/$USERNAME/activation_additions
COPY --chown=$USERNAME:$USER_GID . .
RUN pip install --no-cache-dir -e '.[dev]'


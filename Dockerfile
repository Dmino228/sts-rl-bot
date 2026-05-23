FROM python:3.12-slim-bullseye

# Avoid interactive prompts during package installations
ENV DEBIAN_FRONTEND=noninteractive

# Install base dependencies to retrieve Adoptium package repository
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-transport-https \
    ca-certificates \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Add Eclipse Temurin (Adoptium) official package repository
RUN wget -qO - https://packages.adoptium.net/artifactory/api/gpg/key/public | gpg --dearmor | tee /etc/apt/trusted.gpg.d/adoptium.gpg > /dev/null \
    && echo "deb https://packages.adoptium.net/artifactory/deb bookworm main" | tee /etc/apt/sources.list.d/adoptium.list

# Install Adoptium OpenJDK 8 JRE and all necessary native X11, Mesa rendering, and audio shim packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    temurin-8-jre \
    xvfb \
    mesa-utils \
    libglu1-mesa \
    libgl1-mesa-glx \
    x11-xserver-utils \
    libxcursor1 \
    libxrandr2 \
    libxxf86vm1 \
    libopenal1 \
    && rm -rf /var/lib/apt/lists/*

# System environment variable overrides to enforce Mesa software rendering headlessly
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV MESA_GL_VERSION_OVERRIDE=3.3
ENV MESA_GLSL_VERSION_OVERRIDE=330
ENV GALLIUM_DRIVER=softpipe
ENV DISPLAY=:99

WORKDIR /workspace

# Copy Python requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
    && pip uninstall -y opencv-python opencv-python-headless

# Copy all project code
COPY . .

# Create a Java runner shim inside SlayTheSpire/jre/bin to bypass ModTheSpire's Steam JRE check on Linux
RUN rm -rf /workspace/SlayTheSpire/jre \
    && mkdir -p /workspace/SlayTheSpire/jre/bin \
    && printf '#!/bin/bash\nexec java "$@"\n' > /workspace/SlayTheSpire/jre/bin/java \
    && chmod +x /workspace/SlayTheSpire/jre/bin/java

# Set execution rights on bootstrap wrapper script
RUN chmod +x run_docker_cluster.sh

# Set the default entrypoint
ENTRYPOINT ["/workspace/run_docker_cluster.sh"]

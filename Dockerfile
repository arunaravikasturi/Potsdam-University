FROM ubuntu:latest
ENV DEBIAN_FRONTEND=noninteractive

# Install required packages
RUN apt-get update && \
    apt-get install -y python3 python3-numpy python3-statsmodels nano git && \
    rm -rf /var/lib/apt/lists/*

# Ensure the repository folder exists at /tmp
RUN mkdir -p /tmp/exampleRepository

# Keep the container alive by default
CMD ["tail", "-f", "/dev/null"]

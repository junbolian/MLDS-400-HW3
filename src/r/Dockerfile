# src/r/Dockerfile
FROM r-base:4.3.3

WORKDIR /app

# Copy R scripts
COPY src/r/install_packages.R /app/src/r/install_packages.R
COPY src/r/train.R /app/src/r/train.R

# Create dirs
RUN mkdir -p /app/src/data /app/artifacts

# Install packages
RUN Rscript /app/src/r/install_packages.R

# Default command
CMD ["Rscript", "/app/src/r/train.R"]

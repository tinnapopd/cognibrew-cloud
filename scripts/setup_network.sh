#!/usr/bin/env bash

set -x
set -eo pipefail

if ! [ -x "$(command -v docker)" ]; then
    echo >&2 "Error: Docker is not installed."
    exit 1
fi

>&2 echo "Cognibrew Cloud Setup"

# Create external Docker network if it doesn't exist
if ! docker network inspect cognibrew-cloud >/dev/null 2>&1; then
    >&2 echo "Creating Docker network: cognibrew-cloud"
    docker network create cognibrew-cloud
else
    >&2 echo "Docker network 'cognibrew-cloud' already exists"
fi

>&2 echo "Setup complete"

#!/bin/bash

usage() {
    echo "Usage: $0 <version> <os>"
    echo "  version: 6.1 or 6.2"
    echo "  os: ubuntu20, ubuntu22, or ubuntu24"
    exit 1
}

if [ "$#" -ne 2 ]; then
    usage
fi

VERSION=$1
OS=$2

if [[ "$VERSION" != "6.1" && "$VERSION" != "6.2" ]]; then
    echo "Error: Only versions 6.1 and 6.2 supported at this time"
    usage
fi

if [[ "$OS" != "ubuntu20" && "$OS" != "ubuntu22" && "$OS" != "ubuntu24" ]]; then
    echo "Error: OS must be ubuntu20, ubuntu22, or ubuntu24"
    usage
fi

case "${OS}_${VERSION}" in
    "ubuntu20_6.1")
        BASE_IMAGE="rocm/rocm-build-ubuntu-20.04:6.1"
        ;;
    "ubuntu20_6.2")
        BASE_IMAGE="rocm/dev-ubuntu-20.04:6.2-complete"
        ;;
    "ubuntu22_6.1")
        BASE_IMAGE="rocm/rocm-build-ubuntu-22.04:6.1"
        ;;
    "ubuntu22_6.2")
        BASE_IMAGE="rocm/dev-ubuntu-22.04:6.2-complete"
        ;;
    "ubuntu24_6.2")
        BASE_IMAGE="rocm/dev-ubuntu-24.04:6.2-complete"
        ;;
    *)
        echo "Error: Unsupported OS and version combination"
        exit 1
        ;;
esac

echo "Pulling Docker image: $BASE_IMAGE"
docker pull $BASE_IMAGE

fake_group="$PWD/group"
fake_passwd="$PWD/passwd"

getent group > "${fake_group}"
getent passwd > "${fake_passwd}"

docker run -ti \
    -e ROCM_VERSION=${VERSION} \
    -e CCACHE_DIR=$HOME/.ccache \
    -e CCACHE_ENABLED=true \
    -e DOCK_WORK_FOLD=/src \
    -w /src \
    -v $PWD:/src \
    --mount="type=bind,src=${fake_group},dst=/etc/group,readonly" \
    --mount="type=bind,src=${fake_passwd},dst=/etc/passwd,readonly" \
    -v ${HOME}/.ccache:${HOME}/.ccache \
    -u $(id -u):$(id -g) \
    ${BASE_IMAGE} bash

rm "${fake_group}" "${fake_passwd}"
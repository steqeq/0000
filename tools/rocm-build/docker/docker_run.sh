#!/bin/bash

usage() {
    echo >&2 "Usage: $0 <version> <os>"
    echo >&2 "  version: 6.1 or 6.2"
    echo >&2 "  os: ubuntu20, ubuntu22, or ubuntu24"
    echo >&2 "Supported combinations:"
    echo >&2 "  ubuntu20_6.1, ubuntu20_6.2"
    echo >&2 "  ubuntu22_6.1, ubuntu22_6.2"
    echo >&2 "  ubuntu24_6.2"
    exit 1
}

if [ "$#" -ne 2 ]; then
    usage
fi

VERSION=$1
OS=$2

case "${OS}_${VERSION}" in
    ("ubuntu20_6.1")
        BASE_IMAGE="rocm/rocm-build-ubuntu-20.04:6.1"
        ;;
    ("ubuntu20_6.2")
        BASE_IMAGE="rocm/rocm-build-ubuntu-20.04:6.2"
        ;;
    ("ubuntu22_6.1")
        BASE_IMAGE="rocm/rocm-build-ubuntu-22.04:6.1"
        ;;
    ("ubuntu22_6.2")
        BASE_IMAGE="rocm/rocm-build-ubuntu-22.04:6.2"
        ;;
    ("ubuntu24_6.2")
        BASE_IMAGE="rocm/rocm-build-ubuntu-24.04:6.2"
        ;;
    (*)
        echo >&2 "Error: Unsupported OS and version combination"
        usage
        ;;
esac

echo "Pulling Docker image: $BASE_IMAGE"
if ! docker pull $BASE_IMAGE; then
    echo >&2 "Error: Failed to pull Docker image"
    exit 1
fi

fake_passwd="$PWD/passwd"
fake_shadow="$PWD/shadow"
fake_group="$PWD/group"

getent passwd > "${fake_passwd}"
getent group > "${fake_group}"
sed 's/:[^:]*:/:x:/' "${fake_passwd}" > "${fake_shadow}"

mkdir -p ${HOME}/.ccache

docker_exit_code=0
docker run -ti \
    -e ROCM_VERSION=${VERSION} \
    -e CCACHE_DIR=/.ccache \
    -e CCACHE_ENABLED=true \
    -e DOCK_WORK_FOLD=/src \
    -w /src \
    -v $PWD:/src \
    --mount="type=bind,src=${fake_passwd},dst=/etc/passwd,readonly" \
    --mount="type=bind,src=${fake_shadow},dst=/etc/shadow,readonly" \
    --mount="type=bind,src=${fake_group},dst=/etc/group,readonly" \
    -v ${HOME}/.ccache:/.ccache \
    -u $(id -u):$(id -g) \
    ${BASE_IMAGE} bash || docker_exit_code=$?

rm "${fake_passwd}" "${fake_shadow}" "${fake_group}"

exit $docker_exit_code
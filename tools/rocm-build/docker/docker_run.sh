#!/bin/bash

usage() {
    echo >&2 "Usage: $0 <os> [version]"
    echo >&2 "  os: ubuntu20, ubuntu22, or ubuntu24"
    echo >&2 "  version: 6.1 or 6.2 (default: 6.2)"
    echo >&2 "Supported combinations:"
    echo >&2 "  ubuntu20_6.1, ubuntu20_6.2"
    echo >&2 "  ubuntu22_6.1, ubuntu22_6.2"
    echo >&2 "  ubuntu24_6.2"
    exit 1
}

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    usage
fi

OS=$1
VERSION=${2:-6.2}

OS_VERSION=$(echo $OS | sed 's/[^0-9]*//g')

BASE_IMAGE="rocm/rocm-build-ubuntu-$OS_VERSION.04:$VERSION"

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
echo "$(id -un):*:19110:0:99999:7:::" > ${fake_shadow}

mkdir -p ${HOME}/.ccache

docker_exit_code=0
docker run -ti \
    -e ROCM_VERSION=${ROCM_VERSION} \
    -e CCACHE_DIR=$HOME/.ccache \
    -e CCACHE_ENABLED=true \
    -e DOCK_WORK_FOLD=/src \
    -w /src \
    -v $PWD:/src \
    --mount="type=bind,src=${fake_passwd},dst=/etc/passwd,readonly" \
    --mount="type=bind,src=${fake_shadow},dst=/etc/shadow,readonly" \
    --mount="type=bind,src=${fake_group},dst=/etc/group,readonly" \
    -v ${HOME}/.ccache:${HOME}/.ccache \
    -v ${HOME}:/home/$(id -un) \
    -u $(id -u):$(id -g) \
    ${BASE_IMAGE} /bin/bash -c "
        mkdir -p /home/$(id -un)
        chown $(id -u):$(id -g) /home/$(id -un)
        export HOME=/home/$(id -un)
        bash
    " || docker_exit_code=$?

rm "${fake_passwd}" "${fake_shadow}" "${fake_group}"

exit $docker_exit_code

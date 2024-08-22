#!/bin/bash
BASE_IMAGE="rocm/rocm-build-ubuntu-24.04:6.2"

echo "Pulling Docker image: $BASE_IMAGE"
if ! docker pull $BASE_IMAGE; then
    echo >&2 "Error: Failed to pull Docker image"
    exit 1
fi

fake_passwd="$PWD/passwd"
fake_shadow="$PWD/shadow"
fake_group="$PWD/group"

sudo getent passwd > "${fake_passwd}"
sudo getent group > "${fake_group}"
sudo getent shadow > "${fake_shadow}"

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
    --cap-add SYS_ADMIN \
    --cap-add DAC_READ_SEARCH \
    ${BASE_IMAGE} bash || docker_exit_code=$?

rm "${fake_passwd}" "${fake_shadow}" "${fake_group}"

exit $docker_exit_code

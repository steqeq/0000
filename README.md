# AMD ROCm™

This repository contains the manifest file for ROCm™ releases, changelogs, and
release information. The file default.xml contains information for all
repositories and the associated commit used to build the current ROCm release.

The default.xml file uses the repo Manifest format.

The develop branch of this repository contains content for the next
ROCm major or minor release. 

ROCm is versioned centrally per ROCm release and per individual component. ROCm components follow semantic versioning for the individual ROCm projects. For example, HIP and rocBLAS follow semantic versionig. ROCm releases do not follow semantic versioning. ROCm deviates from semantic versioning in the behavior of PATCH releases. ROCm is version
numbered based on MAJOR.MINOR.PATCH where
 - MAJOR version is a incompatible API change has occured in a ROCm component
 - MINOR version is the addition functionality in a backwards compatible manner.
 - PATCH version is the addition of functionality or bug fixes in a backwards compatible manner. Automatic upgrades to patch number releases are not enabled by default.

Derived from [semver](https://semver.org/).

## How to build documentation via Sphinx

```bash
cd docs

pip3 install -r .sphinx/requirements.txt

python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

## Older ROCm™ Releases

For release information for older ROCm™ releases, refer to
[CHANGELOG](./CHANGELOG.md).

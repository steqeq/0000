# ROCm Backward Compatibility

## Definitions

### Deprecation

### Major Release

ROCm components follow a semantic versioning described at https://semver.org.

ROCm releases have a X.Y.Z (major minor patch) versioning scheme similar to semver where:
- X is years since ROCm first released
- Y is month or bi-monthly period of year
- Z is patch release

For example, in ROCm 5.4.1, the major version number is 5, the minor is 4, and the patch is 1.

## Policy

### Libraries and Framework

There is no strong backward compatibility requirements for libraries and framework although AMD tries to give a couple of minor release heads-up when a change is scheduled.

### HIP Runtime

Within a major release of ROCm, the HIP runtime API will be backward compatible. If a breakage is scheduled, it will be announced two minor releases before it happens and will only happens at the next major release.

For example, if an API is planned to be changed for ROCm 6.0, it will be announced when ROCm 5.6 and 5.7 are released.

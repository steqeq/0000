---
main_title: Governance Model
main_body: As the ROCm governance model gets defined, it will be documented here.
slug: governance_doc
---

# Governance Model

## Scope

ROCm is a software stack made up of a collection of drivers, development tools, and APIs that enable GPU programming from low-level kernel to end-user applications.

Components of ROCm that are inherited from external projects (such as [LLVM][LLVM], [Kernel driver][Kernel Driver], etc) shall continue to follow their existing governance model and code of conduct.

All other components of ROCm shall be goverened by this document.

## Governance

ROCm is led and managed by AMD.  We welcome contributions from the community.  Maintainers will review and approve changes into ROCm.

## Roles

* **Maintainers** are responsible for their designated component and repositories.
* **Contributors** provide input and suggested changes to the existing components, as well as help identify and fix issues, by submitting Pull Requests (PR).

### Maintainers

Maintainers are appointed by AMD. They are able to approve changes and can commit to the repositories. As with everyone else, they must use Pull Requests.

The list of maintainers are available in each repository, defined in the GitHub CODEOWNERS file.  Each repository will have different maintainers specified.

### Contributors

Everyone else is a contributor.  The community is encouraged to contribute to ROCm in several ways:

* Post questions/solutions on our [GitHub discussion forums][Github forums] to help out other community members
* File an issue report on [GitHub Issues][Github issues] to notify us of a bug
* Improve our documentation by submitting a pull request (PR) to our [documentation repository][Documentation Github]
* Improve the code base by submitting a PR to the component in GitHub, for smaller/contained changes
* Suggest larger features in the Ideas category on [GitHub discussion forum][Github forums]

For further details, please see our [contribution guidelines][Contribution guidelines].

## Code of Conduct

For AMD components of ROCm which are hosted on GitHub, all users must abide by the [GitHub community guidelines][Github community guidelines] and the [GitHub community code of conduct][Github community code of conduct].

[Github forums]: https://github.com/RadeonOpenCompute/ROCm/discussions
[Github issues]: https://github.com/RadeonOpenCompute/ROCm/issues
[Documentation GitHub]: https://github.com/RadeonOpenCompute/ROCm
[Github community guidelines]: https://docs.github.com/en/site-policy/github-terms/github-community-guidelines
[Github community code of conduct]: https://docs.github.com/en/site-policy/github-terms/github-community-code-of-conduct
[Contribution guidelines]: contributing.md
[Kernel Driver]: https://github.com/RadeonOpenCompute/ROCK-Kernel-Driver
[LLVM]: https://github.com/RadeonOpenCompute/llvm-project

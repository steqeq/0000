---
main_title: Contributing to ROCm
main_body: This is the high level contributing description.  Each component of ROCm can and should detail their own contributing.md which goes into detail about how community members can contribute to their component.
slug: main_constributing_doc
---

# General Contribution Guidelines

AMD values and encourages contributions to our code and documentation.

This document captures the high-level general contribution guidelines and workflow.  Each component of ROCm should describe an in-depth contribution guide following their particular style and methods.  

## Scope

ROCm is a software stack made up of a collection of drivers, development tools, and APIs that enable GPU programming from low-level kernel to end-user applications.

Components of ROCm that are inherited from external projects (such as [LLVM][LLVM], [Kernel driver][Kernel Driver], etc) shall continue to follow their existing contribution guidelines and workflow.

All other components of ROCm shall follow the workflow described by this document.

## Development Workflow

ROCm is currently using the GitHub platform for code hosting, collaboration and managing version control. All development will be done using the standard GitHub workflow of pull requests.

### Issue Tracking

Issues are filed on GitHub Issues.

* Search if issue already exists in [GitHub Issues list][Github issues].
  * Use your best judgement.  If you believe your issue is the same as another, then you can just upvote the issue and comment/post to provide more details as to how you reproduced it.
  * If you're not absolutely sure if the issue is the same, err on the side of caution and just file the issue.  You can add a comment saying it might be the same as another issue number.  Our team will assess and close as duplicate, if it is in fact a duplicate.
  * If issue does NOT exist, use the template to file a new issue.
* When filing the issue, please be sure to provide as much information as possible, including the output of the scripts to collect information about the configuration.  This will help reduce the churn required to understand how to reproduce the issue.
* Check the issue regularly as there may be additional questions or information required to successfully reproduce the issue.

### Pull Request Process

Our repositories follow a workflow where all changes go into the **develop** branch. This branch serves as an integration branch for new code. Pull requests should follow the general flow below.  A particular repository may include additional steps.  Please refer to each repository's PR process for the most detailed steps.

* Identify issue to fix
* Target the **develop** branch for integration
* Ensure code builds successfully
* Ensure change meets acceptance criteria
* Each component has a suite of test cases to run.  Please include the log of the successful test run in your PR
* Do not break existing test cases
* New functionality will only be merged with new unit tests
* Tests must have good code coverage
* Submit the pull request and work with reviewer/maintainer to get PR approved
* Once approved, the PR will be brought onto internal CI systems and may be merged into the component at opportune intervals, coordinated by the maintainer
* You will be informed on the PR once the change is committed.

**IMPORTANT:** By creating a pull request, you agree to allow your contribution to be licensed by the project owners under the terms of the license.txt in the corresponding repository.  Each repository may use a different license.  You can lookup the license on this [summary page][ROCm licenses].

### New Feature Development

Proposed new features should be discussed actively on the [GitHub Discussion forum][Github forums], Ideas category.  Maintainers can provide direction and feedback on feature development.

### Testing

All PRs need to pass an existing test suite.  Please include a log of a successful test run with your PR.

If your PR includes a new feature, an application or test must be provided so that we can test that the feature works & continues to be valid in the future.

For the time being, pull requests will be run on internal CI systems to ensure functionality and robustness.

### Documentation

Changes to the main ROCm documentation can be submitted on the [Documentation GitHub][Documentation GitHub]. The documentation must be updated for any new feature or API update.

Each repository may use a different documentation method or style.  Please follow the documentation process for each repository.

## Future Development Workflow

The current ROCm development workflow is GitHub based.  However, ROCm can move to a different platform.  If the platform does change, the tools and links may change, but the general development workflow shall remain similar to what is currently described above.  If such a change is to occur, please expect that this document will be updated to describe the new tools and links.  

[Github forums]: https://github.com/RadeonOpenCompute/ROCm/discussions
[Github issues]: https://github.com/RadeonOpenCompute/ROCm/issues
[Documentation GitHub]: https://github.com/RadeonOpenCompute/ROCm
[ROCm licenses]: https://rocm.docs.amd.com/en/latest/release/licensing.html
[Kernel Driver]: https://github.com/RadeonOpenCompute/ROCK-Kernel-Driver
[LLVM]: https://github.com/RadeonOpenCompute/llvm-project

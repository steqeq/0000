<head>
  <meta charset="UTF-8">
  <meta name="description" content="Contributing to ROCm">
  <meta name="keywords" content="ROCm, contributing, contribute, maintainer, contributor">
</head>

# Contributing to the ROCm documentation

The ROCm documentation, like all of ROCm, is open source and available on GitHub. You can contribute to the ROCm documentation by forking the appropriate repository, making your changes, and opening a pull request.

To provide feedback on the ROCm documentation, including submitting an issue or suggesting a feature, see [Providing feedback about the ROCm documentation](./feedback.md).

## The ROCm repositories

The repositories for ROCm and for all ROCm components are available on GitHub.

| Module | Documentation location |
| --- | --- |
| ROCm framework | [https://github.com/ROCm/ROCm/tree/develop/docs](https://github.com/ROCm/ROCm/tree/develop/docs) |
| ROCm installer for Linux | [https://github.com/ROCm/rocm-install-on-linux/tree/develop/docs](https://github.com/ROCm/rocm-install-on-linux/tree/develop/docs) |
| ROCm HIP SDK installer for Windows |  [https://github.com/ROCm/rocm-install-on-windows/tree/develop/docs](https://github.com/ROCm/rocm-install-on-windows/tree/develop/docs) |

Individual components have their own repositories with their own documentation in their own `docs` directories.

The `docs` directories across ROCm are structured as follows:

| Directory name | Documentation type |
|-------|----------|
| `install` | Installation instructions, build instructions, and prerequisites |
| `conceptual` | Important concepts |
| `how-to` | How to implement specific use cases |
| `tutorials` | Tutorials |
| `reference` | API references |

## Editing and adding to the documentation

The ROCm documentation is written in reStructuredText (rst) and Github-flavoured Markdown, and follows the [Google developer documentation style guide](https://developers.google.com/style/highlights). reStructuredText is preferred when adding content to the documentation.

To edit or add to the documentation:

1. Fork the repository you want to add to or edit.
2. Clone your fork locally.
3. Create a new local branch cut from the `develop` branch of the repository.
4. Make your changes to the documentation.

5. Optionally, build the documentation locally before creating a pull request by running the following commands from within the `docs` directory:

    ```bash
     pip3 install -r sphinx/requirements.txt  # You only need to run this command once
     python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
     ```

    The output files will be located in the `docs/_build` directory. Open `docs/_build/html/index.html` to view the documentation.

    For more information on the build process, see [Building documentation](building.md).

    For more information on ROCm build tools, see [Documentation toolchain](toolchain.md).
6. Push your changes. A GitHub link will be returned in the output of the `git push` command. Open this link in a browser to create the pull request.

    The documentation is built as part of the checks on pull requests. Always verify that the documentation has been successfully built and that changes are rendered properly.

    Spell checking and linting are performed on pull requests. New words or acronyms can be added to the [wordlist file](https://github.com/ROCm/rocm-docs-core/blob/develop/.wordlist.txt) as needed.

See the GitHub documentation for information on how to fork and clone a repository, and how to create and push a branch.

```{important}
By creating a pull request (PR), you agree to allow your contribution to be licensed under the terms of the
LICENSE.txt file in the corresponding repository. Different repositories can use different licenses.
```

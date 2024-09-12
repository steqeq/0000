<head>
  <meta charset="UTF-8">
  <meta name="description" content="Contributing to ROCm">
  <meta name="keywords" content="ROCm, contributing, contribute, maintainer, contributor">
</head>

# Contribute to ROCm documentation

The documentation repositories for ROCm and for all ROCm projects are available on GitHub under the ROCm organization at [https://github.com/ROCm](https://github.com/ROCm).

Documentation for ROCm and for all ROCm projects is located in the `docs` directory of their repositories.

The main ROCm repository is [https://github.com/ROCm/ROCm](https://github.com/ROCm/ROCm).

The repository for ROCm installation on Linux is [https://github.com/ROCm/rocm-install-on-linux](https://github.com/ROCm/rocm-install-on-linux).

The repository for HIP SDK installation on Windows is [https://github.com/ROCm/rocm-install-on-windows](https://github.com/ROCm/rocm-install-on-windows).

The repositories for all other ROCm projects are findable through a search under [The ROCm organization](https://github.com/ROCm).

You can contribute to the ROCm documentation by participating in discussion, reporting issues, and adding or editing the documentation directly.

## Participate in discussions through GitHub Discussions

You can ask questions, view announcements, suggest new features, and communicate with other members of the community through [GitHub Discussions](https://github.com/ROCm/ROCm/discussions).

## Submit issues through GitHub Issues

You can submit issues through [GitHub Issues](https://github.com/ROCm/ROCm/issues).

Before creating a new issue, search to see if the same issue has already been logged. If same issue already exists, upvote the issue, and comment or post to provide any additional details you might have.

If you find an issue that is similar, open your issue, then add a comment that includes the issue number of the similar issue, and a link to the issue.

Always provide as much information as possible when creating a new issue. This helps reduce the time required to reproduce the issue.

Check your issue regularly for any requests for additional information.

## Edit or add to the documentation directly

And you can edit and add to the documentation by forking a ROCm repository and submitting a pull request.

Documentation for ROCm and for all ROCm projects is located in the `docs` directory of the repository.

To edit or add to ROCm or a ROCm project's documentation:

1. Fork the repository of the documentation you want to add to or edit.
2. Clone your fork locally.
3. Create a new local branch cut from the `develop` branch of the repository.
4. Make your changes to the documentation.

    Documentation for ROCm and ROCm projects is written in reStructuredText (rst) and Markdown. File names are in dash-case. ROCm documentation follows the [Google developer documentation style guide](https://developers.google.com/style/highlights) and is structured according to the [the Diàtaxis model](https://diataxis.fr/how-to-use-diataxis/).
5. [Optional] Build the documentation locally before creating a pull request by running the following commands from within the `docs` directory:

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

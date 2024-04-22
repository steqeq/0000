import re

from util.release_data import ReleaseLib
from util.defaults import TEMPLATES, PROCESSORS

TEMPLATES['MIVisionX'] = (
    (
        r"## MIVisionX (?P<lib_version>\d+\.\d+(?:\.\d+))"
        r"( \(Unreleased\))?"
        r"\n"
        r"(?P<body>(?:(?!## ).*(?:(?!\n## )\n|(?=\n## )))*)"
    )
)


def mivisionx_processor(data: ReleaseLib, template: str, _, __) -> bool:
    """Processor for MIVisionX releases."""
    changelog = data.repo.get_contents("CHANGELOG.md", data.commit)
    changelog = changelog.decoded_content.decode()
    pattern = re.compile(template)
    match = pattern.search(changelog)
    lib_version  = match["lib_version"]
    data.message = (
        f"MIVisionX for ROCm"
        f" {data.full_version}"
    )

    readme = data.repo.get_contents("README.md", data.commit)
    readme = readme.decoded_content.decode()
    dependency_map = readme[readme.find("## MIVisionX Dependency Map"):]
    data.lib_version = lib_version
    data.notes = f"""{match["body"]}
{dependency_map}
"""
    
    change_pattern = re.compile(
        r"^#+ +(?P<type>[^\n]+)$\n*(?P<change>(^(?!#).*\n*)*)",
        re.RegexFlag.MULTILINE
    )
    for match in change_pattern.finditer(data.notes):
        data.data.changes[match["type"]] = match["change"]
    
    return True

PROCESSORS['MIVisionX'] = mivisionx_processor

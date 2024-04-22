import re

from util.release_data import ReleaseLib
from util.defaults import TEMPLATES, PROCESSORS

TEMPLATES['hipfort'] = (
    (
        r"## hipfort (?P<lib_version>\d+\.\d+(?:\.\d+))?"
        r"(?P<for_rocm> for ROCm )?"
        r"(?P<rocm_version>(?(for_rocm)\d+\.\d+(?:\.\d+)?|.*))?"
        r"( \(Unreleased\))?"
        r"\n"
        r"(?P<body>(?:(?!## ).*(?:(?!\n## )\n|(?=\n## )))*)"
    )
)


def hipfort_processor(data: ReleaseLib, template: str, _, __) -> bool:
    """Processor for releases."""
    changelog = data.repo.get_contents("CHANGELOG.md", data.commit)
    changelog = changelog.decoded_content.decode()
    pattern = re.compile(template)
    match = pattern.search(changelog)
    lib_version  = match["lib_version"]
    data.message = (
        f"hipfort for ROCm"
        f" {data.full_version}"
    )

    data.lib_version = lib_version
    data.notes = f"""{match["body"]}"""
    
    change_pattern = re.compile(
        r"^#+ +(?P<type>[^\n]+)$\n*(?P<change>(^(?!#).*\n*)*)",
        re.RegexFlag.MULTILINE
    )
    for match in change_pattern.finditer(data.notes):
        data.data.changes[match["type"]] = match["change"]
    
    return True

PROCESSORS['hipfort'] = hipfort_processor

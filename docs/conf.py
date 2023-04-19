# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import shutil
shutil.copy2('../CONTRIBUTING.md','./contributing.md')

from rocm_docs import ROCmDocs

linkcheck_timeout = 10
linkcheck_request_headers = {
    r'https://docs.github.com/': {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux i686; rv:112.0) Gecko/20100101 Firefox/112.0'}
}
linkcheck_anchors_ignore = [
    'd90e61', 
    'd1667e113', 
    'd2999e60', 
    'building-from-source', 
    'use-the-rocm-build-tool-rbuild', 
    'use-cmake-to-build-migraphx', 
    'example'
]
linkcheck_ignore = [
    "https://rocmdocs.amd.com/projects/ROCmCC/en/latest/", 
    "https://rocmdocs.amd.com/projects/RVS/en/latest/", 
    "https://rocmdocs.amd.com/projects/amdsmi/en/latest/",
    "https://rocmdocs.amd.com/projects/rdc/en/latest/",
    "https://rocmdocs.amd.com/projects/rocmsmi/en/latest/", 
    "https://rocmdocs.amd.com/projects/roctracer/en/latest/",
    "https://rocmdocs.amd.com/projects/MIGraphX/en/latest/",
    "https://rocmdocs.amd.com/projects/rocprofiler/en/latest/",
    "https://github.com/ROCm-Developer-Tools/HIP-VS/blob/master/README.md",
    r"https://www.amd.com/system/files/.*.pdf"
]

docs_core = ROCmDocs("ROCm Docs 5.6.0 Alpha")
docs_core.setup()
docs_core.disable_main_doc_link()

for sphinx_var in ROCmDocs.SPHINX_VARS:
    globals()[sphinx_var] = getattr(docs_core, sphinx_var)

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import shutil
shutil.copy2('../CONTRIBUTING.md','./contributing.md')

from rocm_docs import ROCmDocs

# working anchors that linkcheck cannot find
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
    # site to be built
    "https://rocmdocs.amd.com/projects/ROCmCC/en/latest/", 
    "https://rocmdocs.amd.com/projects/RVS/en/latest/", 
    "https://rocmdocs.amd.com/projects/amdsmi/en/latest/",
    "https://rocmdocs.amd.com/projects/rdc/en/latest/",
    "https://rocmdocs.amd.com/projects/rocmsmi/en/latest/", 
    "https://rocmdocs.amd.com/projects/roctracer/en/latest/",
    "https://rocmdocs.amd.com/projects/MIGraphX/en/latest/",
    "https://rocmdocs.amd.com/projects/rocprofiler/en/latest/",
    "https://github.com/ROCm-Developer-Tools/HIP-VS/blob/master/README.md",
    "https://rocmdocs.amd.com/projects/HIPIFY/en/develop/",
    # correct links that linkcheck times out on
    r"https://www.amd.com/system/files/.*.pdf",
    "https://www.amd.com/en/developer/aocc.html",
    "https://www.amd.com/en/support/linux-drivers",
    "https://www.amd.com/en/technologies/infinity-hub",
    r"https://bitbucket.org/icl/magma/*",
    "http://cs231n.stanford.edu/"
]

article_pages = [
    {"file":"release/gpu_os_support", "os":["linux"], "author":"", "date":"April 25, 2023", "read-time":"5 min read"},
    {"file":"deploy/linux/index", "os":["linux"], "author":"", "date":"April 25, 2023", "read-time":"5 min read"},
    {"file":"deploy/linux/install_overview", "os":["linux"], "author":"", "date":"April 25, 2023", "read-time":"5 min read"},
    {"file":"deploy/linux/prerequisites", "os":["linux"], "author":"", "date":"April 25, 2023", "read-time":"5 min read"},
    {"file":"deploy/linux/quick_start", "os":["linux"], "author":"", "date":"April 25, 2023", "read-time":"5 min read"},
    {"file":"deploy/linux/install", "os":["linux"], "author":"", "date":"April 25, 2023", "read-time":"5 min read"},
    {"file":"deploy/linux/upgrade", "os":["linux"], "author":"", "date":"April 25, 2023", "read-time":"5 min read"},
    {"file":"deploy/linux/uninstall", "os":["linux"], "author":"", "date":"April 25, 2023", "read-time":"5 min read"},
    {"file":"deploy/linux/package_manager_integration", "os":["linux"], "author":"", "date":"April 25, 2023", "read-time":"5 min read"},
    
    {"file":"deploy/quick_start_windows", "os":["windows"], "author":"", "date":"April 25, 2023", "read-time":"5 min read"},
    {"file":"understand/isv_deployment_win", "os":["windows"], "author":"", "date":"April 25, 2023", "read-time":"5 min read"},

    {"file":"about", "os":["linux", "windows"], "author":"", "date":"April 25, 2023", "read-time":"5 min read"},
    {"file":"deploy", "os":["linux", "windows"], "author":"", "date":"April 25, 2023", "read-time":"5 min read"},
]

docs_core = ROCmDocs("ROCm Docs 5.6.0 Alpha")
docs_core.setup()
docs_core.disable_main_doc_link()
docs_core.set_page_article_info("_build/html/", article_pages)

for sphinx_var in ROCmDocs.SPHINX_VARS:
    globals()[sphinx_var] = getattr(docs_core, sphinx_var)

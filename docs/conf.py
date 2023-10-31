from rocm_docs import ROCmDocs

external_toc_path = "./sphinx/_toc.yml"

docs_core = ROCmDocs("rocm-docs-redirects")
docs_core.setup()

external_projects_current_project = "rocm"

for sphinx_var in ROCmDocs.SPHINX_VARS:
    globals()[sphinx_var] = getattr(docs_core, sphinx_var)

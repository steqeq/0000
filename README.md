# rocm-docs-redirects

Redirects ReadtheDocs Community documentation sites to Business

## Building

```shell
cd docs

pip3 install -r sphinx/requirements.txt

python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

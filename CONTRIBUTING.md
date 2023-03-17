# Contributing to ROCm Docs

## Build the docs using VS Code

One can put together a productive environment to author documentation and also test it locally using VS Code with only a handful of extensions. Even though the extension landscape of VS Code is ever changing, here is one example setup that proved useful at the time of writing. In it, one can change/add content, build a new version of the docs using a single VS Code Task (or hotkey), see all errors/warnings emitted by Sphinx in the Problems pane and immediately see the resulting website show up on a locally serving web server.

### Configuring VS Code

1. Install the following extensions:

   - Python (ms-python.python)
   - Live Server (ritwickdey.LiveServer)

2. Add the following entries in `.vscode/settings.json`

    ```json
    {
    	"liveServer.settings.root": "/.vscode/build/html",
    	"liveServer.settings.wait": 1000,
    	"python.terminal.activateEnvInCurrentTerminal": true
    }
    ```

    The settings in order are set for the following reasons:
    - Sets the root of the output website for live previews. Must be changed alongside the `tasks.json` command.
    - Tells live server to wait with the update to give time for Sphinx to regenerate site contents and not refresh before all is don. (Empirical value)
    - Automatic virtual env activation is a nice touch, should you want to build the site from the integrated terminal.

3. Add the following tasks in `.vscode/tasks.json`

    ```json
    {
    	"version": "2.0.0",
    	"tasks": [
    		{
    			"label": "Build Docs",
    			"type": "process",
    			"windows": {
    				"command": "${workspaceFolder}/.venv/Scripts/python.exe"
    			},
    			"command": "${workspaceFolder}/.venv/bin/python3",
    			"args": [
    				"-m",
    				"sphinx",
    				"-j",
    				"auto",
    				"-T",
    				"-b",
    				"html",
    				"-d",
    				"${workspaceFolder}/.vscode/build/doctrees",
    				"-D",
    				"language=en",
    				"${workspaceFolder}/docs",
    				"${workspaceFolder}/.vscode/build/html"
    			],
    			"problemMatcher": [
    				{
    					"owner": "sphinx",
    					"fileLocation": "absolute",
    					"pattern": {
    						"regexp": "^(?:.*\\.{3}\\s+)?(\\/[^:]*|[a-zA-Z]:\\\\[^:]*):(\\d+):\\s+(WARNING|ERROR):\\s+(.*)$",
    						"file": 1,
    						"line": 2,
    						"severity": 3,
    						"message": 4
    					},
    				},
    				{
    					"owner": "sphinx",
    					"fileLocation": "absolute",
    					"pattern": {
    						"regexp": "^(?:.*\\.{3}\\s+)?(\\/[^:]*|[a-zA-Z]:\\\\[^:]*):{1,2}\\s+(WARNING|ERROR):\\s+(.*)$",
    						"file": 1,
    						"severity": 2,
    						"message": 3
    					}
    				}
    			],
    			"group": {
    				"kind": "build",
    				"isDefault": true
    			}
    		},
    	],
    }
    ```

    _(Implementation detail: two problem matchers were needed to be defined, because VS Code doesn't tolerate some problem information being potentially absent. While a single regex could match all types of errors, if a capture group remains empty (the line number doesn't show up in all warning/error messages) but the `pattern` references said empty capture group, VS Code discards the message completely.)_

4. Configure Python virtual environment (venv)

    - From the Command Palette, run `Python: Create Environment`
      - Select `venv` environment and the `docs/.sphinx/requriements.txt` file. _(Simply pressing enter while hovering over the file from the dropdown is insufficient, one has to select the radio button with the 'Space' key if using the keyboard.)_

5. Build the docs

    - Launch the default build Task using either:
      - a hotkey _(default is 'Ctrl+Shift+B')_ or
      - by issuing the `Tasks: Run Build Task` from the Command Palette.
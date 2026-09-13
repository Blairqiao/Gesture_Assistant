# Single PEP 508 Requirements File and Preserving Dual Platform Entrypoints

We decided to use a single `requirements.txt` file parameterized with PEP 508 environment markers (e.g. `sys_platform == 'darwin'` and `sys_platform == 'win32'`) and retain separate `gesture_assistant.py` and `gesture_assistant_windows.py` scripts.

This approach provides a single uniform `pip install -r requirements.txt` command across all operating systems without platform build failures, while avoiding breaking changes or new module dependencies for users running existing entry points.

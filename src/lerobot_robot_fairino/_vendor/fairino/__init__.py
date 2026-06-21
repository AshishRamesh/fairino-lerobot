# Vendored Fairino Python SDK package marker.
#
# The upstream `fairino` package (FAIR-INNOVATION/fairino-python-sdk, linux/fairino)
# ships as an implicit namespace package (no __init__.py). We add this empty marker
# so the vendored copy resolves as a regular package once `_vendor/` is on sys.path.
#
# `Robot.py` is pure-Python (stdlib only: xmlrpc, socket, ctypes, threading), so this
# import path works on any platform — a live controller is only needed at connect time.

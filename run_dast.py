# This script is a wrapper that runs the main dast.py script
# It handles compatibility with older sklearn versions

import joblib
import sklearn
import types
import sys

# Create a fake sklearn.externals module so the old code still works
# (newer sklearn removed this module, so we recreate it here)
sklearn.externals = types.SimpleNamespace(joblib=joblib)
sys.modules["sklearn.externals"] = sklearn.externals

# Compile the code from dast.py and execute it in the current environment
# This allows dast.py to access the sklearn.externals fix we created above
with open("dast.py") as f:
    code = compile(f.read(), "dast.py", "exec")
    exec(code, globals())

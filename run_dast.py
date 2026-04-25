import joblib
import sklearn
import types
import sys

# fake the old sklearn API BEFORE anything loads
sklearn.externals = types.SimpleNamespace(joblib=joblib)
sys.modules["sklearn.externals"] = sklearn.externals

# now run original script
with open("dast.py") as f:
    code = compile(f.read(), "dast.py", "exec")
    exec(code, globals())

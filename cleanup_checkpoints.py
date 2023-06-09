import os
import shutil

for path in ["keras_checkpoints", "artifacts"]:
    for f in os.listdir(path):
        f = os.path.join(path, f)
        if os.path.isfile(f) or os.path.islink(f):
            os.unlink(f)
        elif os.path.isdir(f):
            shutil.rmtree(f)

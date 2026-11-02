"""run_decode with the patches of PATCHES (env: comma list of module names in this folder) applied right after
perseus.impl.attention is imported (so the env exported by run_decode before perseus._core loads is untouched)."""
import importlib
import importlib.abc
import importlib.machinery
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PATCHES = [p for p in os.environ.get("PATCHES", "").split(",") if p]


class _Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != "perseus.impl.attention":
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        orig = spec.loader.exec_module

        def exec_module(module):
            orig(module)
            for p in PATCHES:
                importlib.import_module(p).install()
        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _Hook())
sys.argv = ["run_decode"] + sys.argv[1:]
runpy.run_module("perseus.examples.gpt2.run_decode", run_name="__main__", alter_sys=True)

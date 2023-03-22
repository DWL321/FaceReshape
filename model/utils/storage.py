import os

from .utils import run
from .log_utils import error, info

models_path = "/dellnas/users/users_share/zhangyu/data/idrfacetrack/*"

def download_pretrained_models(root="~/.idrfacetrack"):
    _root = os.path.expanduser(root)

    if not os.path.exists("/dellnas"):
        error(
            "cannot access /dellnas, please manually copy the pretrained models folder "
            f"`{models_path}` into your home directory `~/.idrfacetrack/`"
        )
        raise SystemExit()

    os.makedirs(_root, exist_ok=True)
    info("copying pretrained models")
    run(f"cp -r {models_path} {_root}")

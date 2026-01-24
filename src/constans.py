from pathlib import Path

from pydantic import BaseModel


class _ProjectPaths(BaseModel):
    project: Path = Path.cwd()
    data: Path = project / "data"
    tmp: Path = project / "tmp"
    logs: Path = project / "logs"
    models: Path = project / "models"


PROJECT_PATHS = _ProjectPaths()

IMG_EXT = ("jpg", "png")
IMG_SHAPE = (256, 256)
COLOR_INT_RANGE = 255.0

BATCH_SIZE = 32

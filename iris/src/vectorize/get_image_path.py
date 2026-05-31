##
## POC PROJECT, 2026
## I.R.I.S
## File description:
## return image found in a dir
##

from pathlib import Path

def get_image_in_path(dir: Path) -> list[Path]:

    image_paths = list(dir.glob("*.jpg"))

    if len(image_paths) == 0:
        return []
    print(f"{len(image_paths)} file found. Start vectorization...\n")
    return image_paths

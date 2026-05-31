##
## POC PROJECT, 2026
## I.R.I.S
## File description:
## batch vectorize
##

from pathlib import Path
from .vectorize_data import vectorize_data
from .get_image_path import get_image_in_path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAPTURES_DIR = PROJECT_ROOT.parent / "captures"
FLAG_VECTOR = "-v"

def main() -> None:

    print(f"Search face in : {CAPTURES_DIR}")
    if not CAPTURES_DIR.exists():
        print(f"No dir named : {CAPTURES_DIR}")
        return

    image_paths = get_image_in_path(CAPTURES_DIR)
    if image_paths == []:
        print(f"No file in : {CAPTURES_DIR}")
        return
    data_base = vectorize_data(image_paths)
    if data_base == {}:
        print("Data base is empty. 0 face as been found")
    print(f"\nFinish -> {len(data_base)} face found.")

    if len(sys.argv) == 2 and sys.argv[1] == FLAG_VECTOR:
        for data in data_base:
            print(data_base[data])
if __name__ == "__main__":
    main()
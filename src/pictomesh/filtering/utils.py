import os

from PIL import Image


def get_image_paths(
    folder: str, extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")
) -> list[str]:
    """
    Creates a list of image file paths from a folder.

    Args:
        folder (str): Path to the input folder containing images.
        extensions (tuple[str, ...]): Allowed file extensions to include.

    Returns:
        List[str]: Sorted list of full image file paths matching the given extensions.
    """

    return sorted(
        [
            os.path.join(folder, filename)
            for filename in os.listdir(folder)
            if filename.lower().endswith(extensions)
        ]
    )


def load_images_from_paths(paths: list[str]) -> list[Image.Image]:
    """
    Loads a list of image paths as PIL Image objects ensuring it's in RGB mode.

    Args:
        paths (List[str]): list of file paths to images.

    Returns:
        List[Image.Image]: List of RGB mode Image object.
    """
    return [Image.open(path).convert("RGB") for path in paths]

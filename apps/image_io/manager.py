import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple

# Set base directory and default input media path
BASE_DIR = Path(__file__).resolve().parents[2]  
DEFAULT_MEDIA_INPUT = BASE_DIR / 'media' / 'input'

class ImageManager:
    '''
    Manages loading, basic preprocessing, and storage of image data and metadata.
    '''

    def __init__(
            self, 
            folder_path: Path = DEFAULT_MEDIA_INPUT, 
            extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png')
            ) -> None:
        '''
        Initializes the ImageManager by loading image paths from a specified folder.

        Params:
            folder_path (Path): Path to the folder containing image files. Defaults to 'media/input'.
            extensions (tuple[str, ...]): Valid image file extensions.
        '''
        self.folder_path = folder_path
        self.extensions = extensions
        self.image_paths = self._get_image_paths()
        self.image_names = self._get_image_name()
        self.images = []
        self.metadata = []

    def _get_image_name(self) -> List[str]:
        '''
        Creates a list of image names without extension in the folder matching the expected extensions.

        Returns:
            List[str]: Sorted list (alphabetically) of image file names without their extensions.
        '''
        return sorted(
            [
                filename.stem
                for filename in self.folder_path.iterdir()
                if filename.suffix.lower() in self.extensions
            ]
        )

    def _get_image_paths(self) -> List[Path]:
        '''
        Creates a list of image file paths from a folder.

        Returns:
            List[Path]: Sorted list (alphabetically) of full image file paths matching the given extensions.
        '''
        return sorted(
            [
                filename
                for filename in self.folder_path.iterdir()
                if filename.suffix.lower() in self.extensions
            ]
        )

    def load_images(self, resize: Tuple[int, int] = None) -> None:
        '''
        Loads and optionally resizes all images in the folder.

        Params:
            resize (tuple[int, int], optional): Target size (width, height) for resizing.
        '''
        for path in self.image_paths:
            img = cv2.imread(str(path))  # Convert path to string for OpenCV
            if img is not None:
                if resize:
                    img = cv2.resize(img, resize)
                self.images.append(img)
                self.metadata.append({'path': str(path), 'shape': img.shape})

    def get_images(self) -> List[np.ndarray]:
        '''
        Returns loaded image arrays.

        Returns:
            List[np.ndarray]: A list of loaded images as NumPy arrays.
        '''
        return self.images

    def get_metadata(self) -> List[dict]:
        '''
        Returns metadata for loaded images.

        Returns:
            List[dict]: A list of metadata dicts containing image path and shape.
        '''
        return self.metadata
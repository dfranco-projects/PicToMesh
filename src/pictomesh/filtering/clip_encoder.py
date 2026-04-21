import os
import torch
import open_clip
from PIL import Image
from utils import get_image_paths, load_images_from_paths


class CLIP:

    def __init__(self, model_name='ViT-B-16-plus-240', pretrained='laion400m_e32') -> None:
        '''
        Initializes the CLIP model and preprocessing.

        Args:
            model_name (str, optional): The name of the CLIP model to use. Default is 'ViT-B-16-plus-240'.
            pretrained (str, optional): The pretrained weights to load. Default is 'laion400m_e32'.
        '''
        # Initialize the device
        self.device = (
            'mps' if torch.backends.mps.is_available() else 
            'cuda' if torch.cuda.is_available() else 
            'cpu'
        )

        # Load model and preprocessing transformations
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model.to(self.device)
        self.model.eval()

    def encode_images(self, image_folder: str) -> dict:
        '''
        Extract image features for all images in a folder using the specified model.

        Args:
            image_folder (str): Path to the folder containing images.

        Returns:
            dict: A dictionary where keys are image names (without extensions) and values are the image feature vectors.
        '''
        image_features = {}

        # Get paths to all images in the folder
        image_paths = get_image_paths(image_folder)

        if not image_paths:
            print('No images found in the folder.')
            return image_features

        # Load the images
        images = load_images_from_paths(image_paths)

        # Process each image
        for img_path, img in zip(image_paths, images):
            # Get the image name without the extension
            img_name = os.path.splitext(os.path.basename(img_path))[0]

            # Preprocess the image: resize, normalize, and add batch dimension
            tensor = self.preprocess(img).unsqueeze(0).to(self.device)

            try:
                # Disable gradient calculation (inference only)
                with torch.no_grad():
                    # Extract feature vector from image tensor
                    img_feat = self.model.encode_image(tensor).float()

                    # Normalize the feature vector to unit length (L2 normalization)
                    img_feat /= img_feat.norm(dim=-1, keepdim=True)

                # Store the feature vector in the dictionary
                image_features[img_name] = img_feat

            except Exception as e:
                print(f'Error processing {img_name}: {e}')

        return image_features

        # Compute pairwise similarities
        similarity_data = []
        for (name1, feat1), (name2, feat2) in combinations(image_features.items(), 2):
            sim = util.pytorch_cos_sim(feat1, feat2).item() * 100
            similarity_data.append((name1, name2, round(sim, 2)))

        # Create and return dataframe
        df = pd.DataFrame(similarity_data, columns=['Image1', 'Image2', 'Similarity'])
        df.sort_values('Similarity', ascending=False, inplace=True)

        return df

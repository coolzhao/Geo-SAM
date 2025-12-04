import torch
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from pathlib import Path
from typing import List, Tuple, Optional
import rasterio
import geopandas as gpd
from shapely.geometry import Polygon

class TreeDetector:
    def __init__(self, model_type: str = "vit_h", device: str = "cuda"):
        """Initialize the TreeDetector with SAM2 model.
        
        Args:
            model_type: Type of SAM2 model to use (vit_h, vit_l, vit_b)
            device: Device to run model on (cuda or cpu)
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        
        # The official sam2 implementation uses a config file and checkpoint path
        # This part may need adjustment based on how the final model is loaded.
        # For now, assuming a similar registry or build function.
        model_cfg = f"configs/sam2.1/sam2.1_hiera_{model_type[-1]}.yaml" # e.g., vit_h -> h
        checkpoint = f"./checkpoints/sam2.1_hiera_{model_type[-1]}.pt"
        
        self.model = build_sam2(model_cfg, checkpoint)
        self.model.to(device)
        self.predictor = SAM2ImagePredictor(self.model)
        
    def load_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess aerial image."""
        with rasterio.open(image_path) as src:
            image = src.read().transpose(1, 2, 0)
            if image.shape[2] > 3:
                image = image[:, :, :3]  # Use only RGB channels
        return image
    
    def fine_tune(self, 
                  train_images: List[str],
                  train_masks: List[str],
                  epochs: int = 10,
                  learning_rate: float = 1e-5):
        """Fine-tune SAM2 on tree examples.
        
        Args:
            train_images: List of paths to training images
            train_masks: List of paths to corresponding tree masks
            epochs: Number of training epochs
            learning_rate: Learning rate for fine-tuning
        """
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            total_loss = 0
            for img_path, mask_path in zip(train_images, train_masks):
                image = self.load_image(img_path)
                mask = np.load(mask_path)
                
                # Embed image
                self.predictor.set_image(image)
                
                # Generate prompts from mask
                points = self._generate_points_from_mask(mask)
                point_labels = np.ones(len(points))
                
                # Get prediction and loss
                masks, iou_predictions, low_res_masks = self.predictor.predict(
                    point_coords=points,
                    point_labels=point_labels,
                    multimask_output=False
                )
                
                # Calculate loss
                loss = self._calculate_loss(masks, mask)
                
                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_images)}")
            
        self.model.eval()
    
    def detect_trees(self, 
                     image_path: str,
                     confidence_threshold: float = 0.93,
                     min_area: int = 100) -> gpd.GeoDataFrame:
        """Detect trees in aerial image and return as GeoDataFrame.
        
        Args:
            image_path: Path to aerial image
            confidence_threshold: Confidence threshold for detection
            min_area: Minimum area for tree detection
            
        Returns:
            GeoDataFrame with tree polygons and confidence scores
        """
        image = self.load_image(image_path)
        self.predictor.set_image(image)
        
        # Generate grid of points for detection
        points = self._generate_grid_points(image.shape[:2])
        
        masks = []
        scores = []
        
        # Process points in batches
        batch_size = 64
        for i in range(0, len(points), batch_size):
            batch_points = points[i:i+batch_size]
            batch_labels = np.ones(len(batch_points))
            
            batch_masks, batch_iou, _ = self.predictor.predict(
                point_coords=batch_points,
                point_labels=batch_labels,
                multimask_output=False
            )
            
            masks.extend(batch_masks)
            scores.extend(batch_iou)
        
        # Convert masks to polygons
        polygons = []
        confidences = []
        
        for mask, score in zip(masks, scores):
            if score > confidence_threshold:
                polygon = self._mask_to_polygon(mask)
                if polygon and polygon.area > min_area:
                    polygons.append(polygon)
                    confidences.append(score)
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame({
            'geometry': polygons,
            'confidence': confidences
        })
        
        return gdf
    
    def _generate_points_from_mask(self, mask: np.ndarray) -> np.ndarray:
        """Generate prompt points from mask centroids."""
        from scipy import ndimage
        labeled_mask, num_features = ndimage.label(mask)
        points = []
        
        for i in range(1, num_features + 1):
            y, x = ndimage.center_of_mass(labeled_mask == i)
            points.append([x, y])
            
        return np.array(points)
    
    def _generate_grid_points(self, shape: Tuple[int, int], 
                             spacing: int = 32) -> np.ndarray:
        """Generate grid of points for detection."""
        y, x = np.mgrid[spacing//2:shape[0]:spacing, 
                       spacing//2:shape[1]:spacing]
        return np.column_stack((x.ravel(), y.ravel()))
    
    def _mask_to_polygon(self, mask: np.ndarray) -> Optional[Polygon]:
        """Convert binary mask to polygon."""
        from skimage import measure
        contours = measure.find_contours(mask, 0.5)
        
        if not contours:
            return None
            
        # Find largest contour
        contour = max(contours, key=len)
        
        # Convert to polygon
        try:
            polygon = Polygon(contour)
            if polygon.is_valid:
                return polygon
        except:
            return None
            
        return None
    
    def _calculate_loss(self, pred_masks: torch.Tensor, 
                       true_mask: np.ndarray) -> torch.Tensor:
        """Calculate loss between predicted and true masks."""
        true_mask = torch.from_numpy(true_mask).float().to(self.device)
        pred_masks = torch.from_numpy(pred_masks).float().to(self.device)
        
        # Dice loss
        intersection = (pred_masks * true_mask).sum()
        union = pred_masks.sum() + true_mask.sum()
        dice_loss = 1 - (2.0 * intersection + 1e-6) / (union + 1e-6)
        
        return dice_loss 
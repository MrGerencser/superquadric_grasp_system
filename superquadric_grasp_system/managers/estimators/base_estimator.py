import numpy as np
from typing import List, Optional, Any
from abc import ABC, abstractmethod

# Import existing utilities
from ...utils.point_cloud_utils import PointCloudProcessor
from ...utils.ros_publisher import ROSPublisher
from ...visualization.main_visualizer import PerceptionVisualizer

class BaseEstimator(ABC):
    """Base class for all pose estimation methods"""
    
    def __init__(self, node, method_config: dict, shared_config: dict):
        self.node = node
        self.logger = node.get_logger()
        self.method_config = method_config
        self.shared_config = shared_config
        
        # Initialize shared components from existing utils
        self.point_cloud_processor = PointCloudProcessor(shared_config)
        self.ros_publisher = ROSPublisher(node, shared_config)
        
        # Initialize visualization if needed
        self.visualizer = None
        if self._needs_visualization():
            try:
                self.visualizer = PerceptionVisualizer()
                self.logger.info("PerceptionVisualizer initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize visualizer: {e}")
        
        # Set visualization components in point cloud processor
        self.point_cloud_processor.set_visualization_components(self.visualizer, self.logger)
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize method-specific components"""
        pass
    
    @abstractmethod
    def process_objects(self, object_point_clouds, object_classes, workspace_cloud=None):
        """Process objects for pose estimation"""
        pass
    
    @abstractmethod
    def is_ready(self) -> bool:
        """Check if estimator is ready"""
        pass
    
    @abstractmethod
    def _needs_visualization(self) -> bool:
        """Determine if visualization is needed"""
        pass
    
    def cleanup(self):
        """Clean up resources"""
        if self.visualizer:
            del self.visualizer
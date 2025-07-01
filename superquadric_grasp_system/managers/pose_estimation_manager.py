from .estimators.icp_estimator import ICPEstimator
from .estimators.superquadric_estimator import SuperquadricEstimator
from .base_manager import BaseManager

class PoseEstimationManager(BaseManager):
    """Factory manager that delegates to specific pose estimation methods"""
    
    def __init__(self, node, config):
        super().__init__(node, config)
        
        self.method = config.get('method', 'superquadric')
        shared_config = config.get('shared', {})
        
        # Create the appropriate estimator with proper config extraction
        if self.method == 'icp':
            # For ICP, merge ICP-specific config with top-level visualization settings
            icp_config = {
                **config.get('icp', {}),  # ICP-specific settings
                **{k: v for k, v in config.items() if k.startswith('enable_') or k.startswith('visualize_')}  # Visualization settings
            }
            self.estimator = ICPEstimator(node, icp_config, shared_config)
        else:
            # For Superquadric, merge superquadric-specific config with top-level visualization settings
            superquadric_config = {
                **config.get('superquadric', {}),  # Superquadric-specific settings
                **{k: v for k, v in config.items() if k.startswith('enable_') or k.startswith('visualize_')}  # Visualization settings
            }
            self.estimator = SuperquadricEstimator(node, superquadric_config, shared_config)
        
        # Common attributes
        self.class_names = {0: 'Cone', 1: 'Cup', 2: 'Mallet', 3: 'Screw Driver', 4: 'Sunscreen'}
        self.graspable_classes = [0, 1, 2, 3, 4]
    
    def initialize(self) -> bool:
        return self.estimator.initialize()
    
    def process_objects(self, object_point_clouds, object_classes, workspace_cloud=None):
        return self.estimator.process_objects(object_point_clouds, object_classes, workspace_cloud)
    
    def is_ready(self) -> bool:
        return self.estimator.is_ready()
    
    def cleanup(self):
        return self.estimator.cleanup()
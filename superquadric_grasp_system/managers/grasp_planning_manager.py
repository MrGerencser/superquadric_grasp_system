import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

from ..utils.grasp_planning.superquadric_grasp_planner import SuperquadricGraspPlanner
from ..utils.grasp_planning.geometric_primitives import Superquadric

class GraspPlanningManager:
    """Manager for superquadric-based grasp planning operations"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        
        # Extract gripper parameters from config
        if config is not None:
            self.gripper_jaw_len = config.get('gripper_jaw_length', 0.041)
            self.gripper_max_open = config.get('gripper_max_opening', 0.08)
            # Debug visualization settings from config
            self.debug_support_test = config.get('enable_support_test_visualization', False)
            self.debug_collision_test = config.get('enable_collision_test_visualization', False)
        else:
            self.gripper_jaw_len = 0.041
            self.gripper_max_open = 0.08
            self.debug_support_test = False
            self.debug_collision_test = False

        self.planner = SuperquadricGraspPlanner(jaw_len=self.gripper_jaw_len, max_open=self.gripper_max_open)
        self.last_planning_results = {}
        
        self.logger.info(f"Initialized with debug - Support: {self.debug_support_test}, Collision: {self.debug_collision_test}")


    def plan_grasps_for_object(self, point_cloud_path: str, superquadric_params: Dict,
                              max_grasps: int = 5) -> Tuple[List[Dict], Optional[Dict]]:
        """
        Plan grasps for a single object with superquadric fitting
        
        Returns:
            Tuple of (all_grasps, best_grasp)
        """
        try:
            shape = superquadric_params['shape']
            scale = superquadric_params['scale'] 
            euler = superquadric_params['euler']
            translation = superquadric_params['translation']
            
            # Use the enhanced planning with best selection
            all_grasps, best_grasp = self.planner.plan_grasps_with_best_selection(
                point_cloud_path, shape, scale, euler, translation, max_grasps
            )
            
            # Store results for later use
            self.last_planning_results = {
                'point_cloud_path': point_cloud_path,
                'superquadric_params': superquadric_params,
                'all_grasps': all_grasps,
                'best_grasp': best_grasp,
                'grasp_count': len(all_grasps)
            }
            
            self.logger.info(f"Planned {len(all_grasps)} grasps, selected best with score: "
                           f"{best_grasp['score']:.6f}" if best_grasp else "None")
            
            return all_grasps, best_grasp
            
        except Exception as e:
            self.logger.error(f"Grasp planning failed: {e}")
            return [], None
    
    def plan_grasps_multi_superquadric(self, point_cloud_path: str, 
                                superquadric_list: List[Dict],
                                debug_support_test: bool = None,
                                debug_collision_test: bool = None) -> List[Dict]:
        """
        Plan grasps for multiple superquadrics and combine results
        """
        use_support_debug = debug_support_test if debug_support_test is not None else self.debug_support_test
        use_collision_debug = debug_collision_test if debug_collision_test is not None else self.debug_collision_test
        
        self.logger.info(f"Planning grasps with debug - Support: {use_support_debug}, Collision: {use_collision_debug}")
        
        all_grasps_combined = []
        
        # Plan grasps for each superquadric
        for sq_idx, sq_params in enumerate(superquadric_list):
            try:
                # Get ALL valid grasps from planner
                # NOTE: plan_grasps() already does support + collision filtering!
                all_valid_grasps = self.planner.plan_grasps(
                    point_cloud_path,
                    sq_params['shape'],
                    sq_params['scale'], 
                    sq_params['euler'],
                    sq_params['translation'],
                    max_grasps=50,  # This parameter is ignored - plan_grasps returns ALL valid
                    visualize_collision_test=use_collision_debug,
                    visualize_support_test=use_support_debug
                )
                
                # Add metadata to each grasp
                for grasp in all_valid_grasps:
                    grasp['sq_index'] = sq_idx
                    all_grasps_combined.append(grasp)
                    
                self.logger.info(f"SQ {sq_idx+1}: Generated {len(all_valid_grasps)} valid grasps")
                
            except Exception as e:
                self.logger.error(f"Failed to plan grasps for SQ {sq_idx+1}: {e}")
                import traceback
                traceback.print_exc()
    
        # Sort by score (highest first) 
        all_grasps_combined.sort(key=lambda x: x.get('score', 0.0), reverse=True)
        
        # Store for visualization
        self.last_planning_results = {
            'point_cloud_path': point_cloud_path,
            'all_grasps': all_grasps_combined,
            'final_grasps': all_grasps_combined,  # Same as all_grasps
            'superquadric_count': len(superquadric_list)
        }
        
        self.logger.info(f"Returning {len(all_grasps_combined)} valid grasps (no artificial reduction)")
        return all_grasps_combined

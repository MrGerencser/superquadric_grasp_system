from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from rclpy.node import Node

class BaseManager(ABC):
    """Base manager with standardized interface"""
    
    def __init__(self, node: Node, config: Dict[str, Any]):
        self.node = node
        self.config = config
        self.logger = node.get_logger()
        self._is_initialized = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize manager resources"""
        
    @abstractmethod
    def is_ready(self) -> bool:
        """Check if manager is ready for operation"""
        
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up manager resources"""
    
    @property
    def initialized(self) -> bool:
        return self._is_initialized
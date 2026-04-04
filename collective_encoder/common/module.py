from abc import ABC
from typing import List, Any

class CEModule(ABC):
    '''
    Module for collective encoder base class.
    Implements functionality common to many modules of the project (e.g. log messages, input checks).
    '''
    def __init__(self,
                 verbose: bool = True,
                 **kwargs):
        self.verbose = verbose
        if self.verbose:
            print("\n\n")
            print("="*80)
            print(f"[Initializing module: {self.__class__.__name__}]")
            print("="*80)
    
    def __post_init__(self):
        print("="*80)
    
    def warn(self, message: str):
        """Logs a warning message."""
        print(f"[{self.__class__.__name__} WARNING]: {message}")
    
    def log_msg(self, message: str):
        """Logs a message if verbosity is enabled."""
        if self.verbose:
            print(f"[{self.__class__.__name__}]: {message}")
    
    def log_list(self, message: str, values: List[Any]):
        """Logs a message with list values if verbosity is enabled."""
        if self.verbose:
            print(f"[{self.__class__.__name__}]: {message}:")
            for v in values:
                print(f"    {v}")
            print(f"[{self.__class__.__name__}]: End - {message}")
            
    
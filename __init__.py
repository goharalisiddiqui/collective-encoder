# Backward compatibility wrapper - DEPRECATED
# Import from the proper package location instead

import warnings

warnings.warn(
    "Importing from root __init__.py is deprecated. "
    "Use 'import collective_encoder' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Import everything from the proper package
from collective_encoder import *
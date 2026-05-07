"""
Timestamp utilities for consistent naming and formatting across the project.
"""
from datetime import datetime


def get_readable_timestamp() -> str:
    """Generate a human-readable timestamp for experiment result directories.
    
    Format: "11-feb-12.45pm" (day-month-hour.minuteam/pm in lowercase)
    
    Returns:
        Timestamp string in readable format
        
    Example:
        >>> ts = get_readable_timestamp()
        >>> # Returns something like: "11-feb-12.45pm"
    """
    return datetime.now().strftime("%d-%b-%H.%M%p").lower()


def get_standard_timestamp() -> str:
    """Generate a standard timestamp for file/log naming.
    
    Format: "20260211_124259" (YYYYMMDD_HHMMSS)
    
    Returns:
        Timestamp string in standard format
        
    Example:
        >>> ts = get_standard_timestamp()
        >>> # Returns something like: "20260211_124259"
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

"""RTSD to Belarusian traffic sign code mapping."""

from typing import Dict, Optional


# RTSD class IDs (0-154) to Belarusian GOST traffic sign codes
RTSD_TO_BELARUSIAN = {
    0: "2.1", 1: "1.23", 2: "1.17", 3: "3.24.1", 4: "7.2.1",
    5: "5.20", 6: "5.16.1", 7: "5.12.1", 8: "3.25.1", 9: "5.33",
    10: "7.15", 11: "2.2", 12: "2.4", 13: "7.13", 14: "4.2.1",
    15: "1.20.3", 16: "1.25", 17: "3.4", 18: "7.3.2", 19: "3.4.1",
    20: "4.1.6", 21: "4.2.3", 22: "4.1.1", 23: "1.33", 24: "5.8.5",
    25: "3.27", 26: "1.15", 27: "4.1.2", 28: "6.3", 29: "7.1.1",
    30: "6.7", 31: "5.8.3", 32: "6.11", 33: "1.19", 34: "5.15",
    35: "7.1.4", 36: "7.8", 37: "1.16.1", 38: "1.11.2", 39: "6.6",
    40: "5.8.1", 41: "6.2", 42: "5.8.1", 43: "3.1", 44: "5.9",
    45: "3.25", 46: "2.3", 47: "3.17", 48: "3.20", 49: "5.3",
    50: "1.14.1", 51: "5.35", 52: "7.7", 53: "1.3", 54: "5.21.1",
    55: "5.21.2", 56: "1.4.1", 57: "1.4.2", 58: "3.31", 59: "3.30",
    60: "1.1", 61: "1.2", 62: "2.5", 63: "3.24", 64: "3.27.1",
    65: "3.6", 66: "5.2", 67: "5.7.1", 68: "5.7.2", 69: "3.33",
    70: "5.31", 71: "5.32", 72: "7.11", 73: "7.12", 74: "7.4",
    75: "7.5", 76: "7.6", 77: "7.9", 78: "7.10", 79: "7.14",
    80: "3.32", 81: "3.33", 82: "6.10.1", 83: "6.10.2", 84: "6.4",
    85: "6.5", 86: "6.8", 87: "6.9", 88: "5.4.1", 89: "5.4.2",
    90: "4.1.3", 91: "4.1.4", 92: "1.5", 93: "1.6", 94: "1.7",
    95: "1.8", 96: "1.9", 97: "1.10", 98: "1.11", 99: "1.11.1",
    100: "1.12", 101: "1.13", 102: "1.14", 103: "1.15.1", 104: "1.15.2",
    105: "1.16", 106: "1.18", 107: "1.20", 108: "1.20.1", 109: "1.20.2",
    110: "1.21", 111: "1.22", 112: "1.24", 113: "1.25.1", 114: "1.26",
    115: "1.27", 116: "1.28", 117: "1.29", 118: "1.30", 119: "1.31",
    120: "1.32", 121: "2.6", 122: "3.2", 123: "3.3", 124: "3.5",
    125: "3.7", 126: "3.8", 127: "3.9", 128: "3.10", 129: "3.11",
    130: "3.12", 131: "3.13", 132: "3.14", 133: "3.15", 134: "3.16",
    135: "3.18", 136: "3.19", 137: "3.21", 138: "3.22", 139: "3.23",
    140: "3.26", 141: "3.28", 142: "3.29", 143: "4.1", 144: "4.1.5",
    145: "4.2", 146: "4.2.2", 147: "5.1", 148: "5.5", 149: "5.6",
    150: "5.10", 151: "5.11", 152: "5.14", 153: "5.17", 154: "5.18",
}


class ClassMapper:
    """Utility class for mapping between class IDs and sign codes.
    
    Provides methods to convert between RTSD class IDs and Belarusian
    traffic sign codes.
    """
    
    @staticmethod
    def id_to_belarusian(class_id: int) -> Optional[str]:
        """Convert RTSD class ID to Belarusian traffic sign code.
        
        Args:
            class_id: RTSD class ID (0-154)
            
        Returns:
            Belarusian GOST traffic sign code or None if not found
        """
        return RTSD_TO_BELARUSIAN.get(class_id)
    
    @staticmethod
    def is_valid_class(class_id: int) -> bool:
        """Check if class ID is valid.
        
        Args:
            class_id: Class ID to validate
            
        Returns:
            True if class_id is valid (0-154)
        """
        return class_id in RTSD_TO_BELARUSIAN
    
    @staticmethod
    def get_all_mappings() -> Dict[int, str]:
        """Get all class ID to Belarusian code mappings.
        
        Returns:
            Dictionary of all mappings
        """
        return RTSD_TO_BELARUSIAN.copy()
    
    @staticmethod
    def get_num_classes() -> int:
        """Get total number of traffic sign classes.
        
        Returns:
            Number of classes (155)
        """
        return len(RTSD_TO_BELARUSIAN)

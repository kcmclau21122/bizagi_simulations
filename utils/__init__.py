"""
Utility functions and helpers for the Bizagi Process Simulator.
"""

from .config import ConfigManager
from .time_utils import (
    format_duration,
    format_duration_for_display,
    format_datetime_duration,
    is_work_time,
    advance_to_work_time,
    get_next_work_datetime,
    get_business_days_between,
    get_working_minutes_between
)
from .helpers import (
    ensure_directory_exists,
    get_file_extension,
    is_valid_file_path,
    generate_unique_id,
    safe_filename,
    truncate_string,
    format_exception,
    log_exception,
    chunks,
    dict_to_pretty_json,
    calculate_file_hash,
    retry,
    get_unique_values,
    dict_exclude_keys,
    dict_include_keys,
    flatten_dict,
    get_system_info,
    validate_email,
    pluralize,
    format_list_as_sentence
)

__all__ = [
    # Configuration
    'ConfigManager',
    
    # Time utilities
    'format_duration',
    'format_duration_for_display',
    'format_datetime_duration',
    'is_work_time',
    'advance_to_work_time',
    'get_next_work_datetime',
    'get_business_days_between',
    'get_working_minutes_between',
    
    # General helpers
    'ensure_directory_exists',
    'get_file_extension',
    'is_valid_file_path',
    'generate_unique_id',
    'safe_filename',
    'truncate_string',
    'format_exception',
    'log_exception',
    'chunks',
    'dict_to_pretty_json',
    'calculate_file_hash',
    'retry',
    'get_unique_values',
    'dict_exclude_keys',
    'dict_include_keys',
    'flatten_dict',
    'get_system_info',
    'validate_email',
    'pluralize',
    'format_list_as_sentence'
]

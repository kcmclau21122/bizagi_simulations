import os
import re
import logging
import json
import sys
import traceback
import hashlib
import random
import string
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, TypeVar, Set

T = TypeVar('T')  # Generic type for type hints

def ensure_directory_exists(directory_path: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logging.info(f"Created directory: {directory_path}")

def get_file_extension(file_path: str) -> str:
    """
    Get the extension of a file path.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File extension (lowercase) with leading dot
    """
    _, ext = os.path.splitext(file_path)
    return ext.lower()

def is_valid_file_path(file_path: str, required_extensions: Optional[List[str]] = None) -> bool:
    """
    Check if a file path is valid and has the required extension.
    
    Args:
        file_path: Path to the file
        required_extensions: List of allowed extensions (with leading dot)
        
    Returns:
        True if the file path is valid, False otherwise
    """
    if not file_path or not os.path.exists(file_path):
        return False
        
    if required_extensions:
        ext = get_file_extension(file_path)
        return ext in required_extensions
        
    return True

def generate_unique_id(prefix: str = "") -> str:
    """
    Generate a unique ID based on time and random characters.
    
    Args:
        prefix: Optional prefix for the ID
        
    Returns:
        Unique ID string
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    random_part = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    return f"{prefix}{timestamp}{random_part}"

def create_directory_if_not_exists(path: str) -> None:
    """
    Create a directory if it doesn't exist.
    
    Args:
        path: Directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def safe_filename(filename: str) -> str:
    """
    Convert a string to a safe filename.
    
    Args:
        filename: The filename to sanitize
        
    Returns:
        A sanitized filename
    """
    # Remove invalid characters
    safe_name = re.sub(r'[\\/*?:"<>|]', '', filename)
    
    # Replace spaces with underscores
    safe_name = safe_name.replace(' ', '_')
    
    # Ensure the filename is not too long
    if len(safe_name) > 255:
        name, ext = os.path.splitext(safe_name)
        safe_name = name[:255-len(ext)] + ext
        
    return safe_name

def truncate_string(text: str, max_length: int = 100, 
                  suffix: str = "...") -> str:
    """
    Truncate a string to a maximum length, adding a suffix if truncated.
    
    Args:
        text: The string to truncate
        max_length: Maximum length of the returned string
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
        
    return text[:max_length-len(suffix)] + suffix

def format_exception(e: Exception) -> str:
    """
    Format an exception into a readable string with traceback.
    
    Args:
        e: The exception
        
    Returns:
        Formatted exception string
    """
    tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
    return ''.join(tb_lines)

def log_exception(e: Exception, message: str = "An error occurred") -> None:
    """
    Log an exception with traceback.
    
    Args:
        e: The exception
        message: Additional message to log
    """
    logging.error(f"{message}: {str(e)}")
    logging.debug(format_exception(e))

def chunks(lst: List[T], chunk_size: int) -> List[List[T]]:
    """
    Split a list into chunks of a given size.
    
    Args:
        lst: The list to split
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def dict_to_pretty_json(data: Dict[str, Any]) -> str:
    """
    Convert a dictionary to pretty JSON.
    
    Args:
        data: Dictionary to convert
        
    Returns:
        Pretty JSON string
    """
    return json.dumps(data, indent=4, sort_keys=True)

def calculate_file_hash(file_path: str) -> str:
    """
    Calculate the MD5 hash of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        MD5 hash as a hexadecimal string
    """
    hash_md5 = hashlib.md5()
    
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
            
    return hash_md5.hexdigest()

def retry(max_attempts: int = 3, delay: float = 1.0,
         exceptions: Tuple[Exception, ...] = (Exception,),
         on_retry: Optional[Callable[[int, Exception], None]] = None) -> Callable:
    """
    Retry decorator for functions that might fail.
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Delay between attempts in seconds
        exceptions: Tuple of exceptions to catch
        on_retry: Function to call on each retry
        
    Returns:
        Decorator function
    """
    import time
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            attempt = 1
            while attempt <= max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        raise
                    if on_retry:
                        on_retry(attempt, e)
                    logging.warning(
                        f"Retry {attempt}/{max_attempts} for {func.__name__} "
                        f"due to {type(e).__name__}: {str(e)}"
                    )
                    time.sleep(delay)
                    attempt += 1
        return wrapper
    return decorator

def get_unique_values(data: List[Dict[str, Any]], key: str) -> List[Any]:
    """
    Get unique values for a key from a list of dictionaries.
    
    Args:
        data: List of dictionaries
        key: Key to extract values for
        
    Returns:
        List of unique values
    """
    return list({item.get(key) for item in data if key in item})

def dict_exclude_keys(d: Dict[str, Any], keys: Set[str]) -> Dict[str, Any]:
    """
    Create a new dictionary excluding specified keys.
    
    Args:
        d: Dictionary to filter
        keys: Set of keys to exclude
        
    Returns:
        New dictionary without excluded keys
    """
    return {k: v for k, v in d.items() if k not in keys}

def dict_include_keys(d: Dict[str, Any], keys: Set[str]) -> Dict[str, Any]:
    """
    Create a new dictionary including only specified keys.
    
    Args:
        d: Dictionary to filter
        keys: Set of keys to include
        
    Returns:
        New dictionary with only included keys
    """
    return {k: v for k, v in d.items() if k in keys}

def flatten_dict(d: Dict[str, Any], parent_key: str = '', 
               separator: str = '.') -> Dict[str, Any]:
    """
    Flatten a nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key for nested dictionaries
        separator: Separator between keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{separator}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, separator).items())
        else:
            items.append((new_key, v))
    return dict(items)

def get_system_info() -> Dict[str, str]:
    """
    Get basic system information.
    
    Returns:
        Dictionary of system information
    """
    import platform
    
    return {
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'system': platform.system(),
        'processor': platform.processor(),
        'machine': platform.machine()
    }

def validate_email(email: str) -> bool:
    """
    Validate an email address.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid, False otherwise
    """
    # Simple regex for email validation
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def pluralize(count: int, singular: str, plural: str = "") -> str:
    """
    Return singular or plural form based on count.
    
    Args:
        count: Count to determine form
        singular: Singular form
        plural: Plural form (if empty, adds 's' to singular)
        
    Returns:
        Appropriate form based on count
    """
    if not plural:
        plural = singular + 's'
        
    return singular if count == 1 else plural

def format_list_as_sentence(items: List[str], 
                          conjunction: str = 'and') -> str:
    """
    Format a list as a comma-separated sentence.
    
    Args:
        items: List of items
        conjunction: Conjunction to use
        
    Returns:
        Formatted string
    """
    if not items:
        return ""
        
    if len(items) == 1:
        return items[0]
        
    return ", ".join(items[:-1]) + f" {conjunction} " + items[-1]

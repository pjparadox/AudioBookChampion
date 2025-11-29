import os

def ensure_dir(directory):
    """
    Ensures that a directory exists. If it does not exist, it is created.

    Args:
        directory (str): The path to the directory to check or create.

    Returns:
        None
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

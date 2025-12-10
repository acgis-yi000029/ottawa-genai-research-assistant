"""
Data Constraints

"""

import functools
from pathlib import Path
from typing import Any, Callable, TypeVar, Union

from app.core.data_paths import monk_paths

F = TypeVar("F", bound=Callable[..., Any])


class DataConstraintError(Exception):
    pass


def validate_monk_path(path: Union[str, Path]) -> Path:
    path_obj = Path(path)
    
    if not monk_paths.validate_monk_path(str(path)):
        raise DataConstraintError(
            f"The data file path must be in the monk directory: {path}. "
            f"Please use monk_paths.get_data_file_path() to get the correct path."
        )
    
    return path_obj


def enforce_monk_directory(func: F) -> F:
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if 'data_file' in kwargs:
            if kwargs['data_file'] is not None:
                validate_monk_path(kwargs['data_file'])
        
        if len(args) > 1 and isinstance(args[1], str):
            validate_monk_path(args[1])
        
        return func(*args, **kwargs)
    
    return wrapper


def validate_data_operation(operation_type: str, file_path: str) -> bool:
    try:
        validate_monk_path(file_path)
        print(f"Data operation verification passed: {operation_type} -> {file_path}")
        return True
    except DataConstraintError as e:
        print(f"Data manipulation violates constraints: {operation_type} -> {file_path}")
        print(f"   Error: {e}")
        return False


class MonkDataValidator:
    
    @staticmethod
    def validate_create_operation(file_path: str) -> None:
        if not validate_data_operation("CREATE", file_path):
            raise DataConstraintError(f"Data creation operation rejected: {file_path}")
    
    @staticmethod
    def validate_write_operation(file_path: str) -> None:
        if not validate_data_operation("WRITE", file_path):
            raise DataConstraintError(f"数Data creation operation rejected: {file_path}")
    
    @staticmethod
    def validate_repository_init(data_file: str) -> None:
        if not validate_data_operation("REPOSITORY_INIT", data_file):
            raise DataConstraintError(f"Repository initialization rejected: {data_file}")
    
    @staticmethod
    def get_recommended_paths() -> dict[str, str]:
        return {
            "User data": monk_paths.USERS_FILE,
            "Document data": monk_paths.DOCUMENTS_FILE,
            "Document block data": monk_paths.CHUNKS_FILE,
            "Dialogue data": monk_paths.CONVERSATIONS_FILE,
            "Message Data": monk_paths.MESSAGES_FILE,
            "Report data": monk_paths.REPORTS_FILE,
            "System Settings": monk_paths.SYSTEM_SETTINGS_FILE,
        }


# 全局验证器实例
data_validator = MonkDataValidator()


def print_data_constraints_info():
    print("\n" + "="*60)
    print("MONK Data directory constraints")
    print("="*60)
    print("All data must be stored in the monk/ directory.")
    print("Use monk_paths to get standard paths")
    print("The repository class has a default Monk path configured.")
    print("\nRecommended path:")
    
    for desc, path in data_validator.get_recommended_paths().items():
        print(f"  📁 {desc}: {path}")
    
    print("="*60 + "\n")


if __name__ != "__main__":
    print_data_constraints_info() 
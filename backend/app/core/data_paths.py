"""
Data Paths Management

"""

from pathlib import Path
from typing import Dict


class MonkDataPaths:
    """monk/ Directory Data Path Manager"""
    
    MONK_BASE_DIR = "monk"
    
    USERS_FILE = f"{MONK_BASE_DIR}/users/users.json"
    DOCUMENTS_FILE = f"{MONK_BASE_DIR}/documents/documents.json"
    CHUNKS_FILE = f"{MONK_BASE_DIR}/documents/chunks.json"
    CONVERSATIONS_FILE = f"{MONK_BASE_DIR}/chats/conversations.json"
    MESSAGES_FILE = f"{MONK_BASE_DIR}/chats/messages.json"
    REPORTS_FILE = f"{MONK_BASE_DIR}/reports/reports.json"
    SYSTEM_SETTINGS_FILE = f"{MONK_BASE_DIR}/system/settings.json"
    
    USERS_DIR = f"{MONK_BASE_DIR}/users"
    DOCUMENTS_DIR = f"{MONK_BASE_DIR}/documents"
    CHATS_DIR = f"{MONK_BASE_DIR}/chats"
    REPORTS_DIR = f"{MONK_BASE_DIR}/reports"
    SYSTEM_DIR = f"{MONK_BASE_DIR}/system"
    VECTOR_DB_DIR = f"{MONK_BASE_DIR}/vector_db"
    
    @classmethod
    def ensure_monk_directories(cls) -> None:
        directories = [
            cls.USERS_DIR,
            cls.DOCUMENTS_DIR,
            cls.CHATS_DIR,
            cls.REPORTS_DIR,
            cls.SYSTEM_DIR,
            cls.VECTOR_DB_DIR,
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_data_file_path(cls, data_type: str) -> str:
        path_mapping = {
            "users": cls.USERS_FILE,
            "documents": cls.DOCUMENTS_FILE,
            "chunks": cls.CHUNKS_FILE,
            "conversations": cls.CONVERSATIONS_FILE,
            "messages": cls.MESSAGES_FILE,
            "reports": cls.REPORTS_FILE,
            "system_settings": cls.SYSTEM_SETTINGS_FILE,
        }
        
        if data_type not in path_mapping:
            raise ValueError(f"Unknown data type: {data_type}")
        
        return path_mapping[data_type]
    
    @classmethod
    def get_directory_path(cls, directory_type: str) -> str:
        directory_mapping = {
            "users": cls.USERS_DIR,
            "documents": cls.DOCUMENTS_DIR,
            "chats": cls.CHATS_DIR,
            "reports": cls.REPORTS_DIR,
            "system": cls.SYSTEM_DIR,
            "vector_db": cls.VECTOR_DB_DIR,
        }
        
        if directory_type not in directory_mapping:
            raise ValueError(f"Unknown directory type: {directory_type}")
        
        return directory_mapping[directory_type]
    
    @classmethod
    def validate_monk_path(cls, path: str) -> bool:
        path_parts = Path(path).parts
        return path_parts[0] == "monk" or "monk" in path_parts
    
    @classmethod
    def get_all_data_files(cls) -> Dict[str, str]:
        return {
            "users": cls.USERS_FILE,
            "documents": cls.DOCUMENTS_FILE,
            "chunks": cls.CHUNKS_FILE,
            "conversations": cls.CONVERSATIONS_FILE,
            "messages": cls.MESSAGES_FILE,
            "reports": cls.REPORTS_FILE,
            "system_settings": cls.SYSTEM_SETTINGS_FILE,
        }


monk_paths = MonkDataPaths()

monk_paths.ensure_monk_directories() 
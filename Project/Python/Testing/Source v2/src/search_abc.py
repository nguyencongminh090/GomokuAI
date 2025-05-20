"""
Abstract Base Classes for search components.
Based on Rapfi's searcher.h.
"""
from abc import ABC, abstractmethod
from typing import Any, Optional # For SearchThread type hint if it's not fully defined

# Forward declaration or placeholder for SearchThread equivalent type
# If SearchThread is defined elsewhere and fully typed, use that.
# For now, using Any or a simple placeholder.
SearchThreadType = Any

class SearchDataBase(ABC):
    """
    Abstract base class for search-specific data.
    Corresponds to Rapfi's SearchData.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def clear_data(self, search_thread_instance: SearchThreadType):
        """
        Clears the states of search data for a new search.
        The `search_thread_instance` provides context, like current search options.
        """
        pass

class SearcherBase(ABC):
    """
    Abstract base class for different search algorithm implementations.
    Corresponds to Rapfi's Searcher.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def make_search_data(self, search_thread_instance: SearchThreadType) -> SearchDataBase:
        """Creates an instance of search data (e.g., ABSearchData) for a search thread."""
        pass

    @abstractmethod
    def set_memory_limit(self, memory_size_kb: int) -> None:
        """Sets the memory size limit for the search (e.g., for TT)."""
        pass

    @abstractmethod
    def get_memory_limit(self) -> int:
        """Gets the current memory size limit in KiB."""
        pass

    @abstractmethod
    def clear(self, thread_pool_instance: Any, clear_all_memory: bool) -> None: # thread_pool_instance type later
        """Clears all searcher states between different games."""
        pass

    @abstractmethod
    def search_main(self, main_search_thread_instance: Any) -> None: # main_search_thread_instance type later
        """Main-thread search entry point."""
        pass

    @abstractmethod
    def search(self, search_thread_instance: SearchThreadType) -> None:
        """Worker-thread search entry point (or main search logic in single-threaded)."""
        pass

    @abstractmethod
    def check_timeup_condition(self) -> bool:
        """Checks if the search has run out of time."""
        pass
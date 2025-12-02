"""Thread-safe Singleton metaclass implementation."""

import threading
from abc import ABCMeta


class SingletonMeta(ABCMeta):
    """
    Thread-safe implementation of Singleton.
    Call setup_singleton() at startup to initialize the lock.
    """

    _instances = {}
    _lock = threading.RLock()

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """

        # Double-checked locking pattern
        if cls not in cls._instances:
            with cls._lock:
                # Check again inside the lock
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]

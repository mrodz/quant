from __future__ import annotations

from typing import TypeVar, Generic
from abc import ABC, abstractmethod
from quant import SessionProvider

IT = TypeVar('IT')
RT = TypeVar('RT')

class PreparedStudy(ABC, Generic[IT, RT]):
    def inputs(self) -> IT:
        ...
    
    def run(self, session: SessionProvider) -> RT:
        ...


class Study(ABC, Generic[IT, RT]):
    @classmethod
    @abstractmethod
    def name(cls) -> str:
        ...

    @abstractmethod
    def prepare_args(self, *args, **kwargs) -> PreparedStudy[IT, RT]:
        ...
        
        
from . import fi_ltm
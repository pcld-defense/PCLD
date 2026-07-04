"""A tiny decorator-based registry for pluggable components.

Datasets, models, and attacks each own a ``Registry`` instance.  New
components register themselves with a decorator, so adding one never requires
editing the runner:

    from pcld.attacks.registry import ATTACKS

    @ATTACKS.register('my_attack')
    def my_attack(model, x, y, *, epsilon, **kw):
        ...

    fn = ATTACKS.get('my_attack')
"""

from typing import Callable, Dict, Generic, Iterator, TypeVar

T = TypeVar('T')


class Registry(Generic[T]):
    """Maps string keys to registered objects (functions, classes, builders).

    Args:
        name: Human-readable registry name, used in error messages.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._items: Dict[str, T] = {}

    def register(self, key: str) -> Callable[[T], T]:
        """Returns a decorator that registers the decorated object under ``key``.

        Args:
            key: The lookup name to register the object under.

        Returns:
            A decorator that stores its argument and returns it unchanged.

        Raises:
            KeyError: If ``key`` is already registered.
        """
        def _decorator(obj: T) -> T:
            if key in self._items:
                raise KeyError(
                    f'{self._name!r} already has an entry named {key!r}.')
            self._items[key] = obj
            return obj

        return _decorator

    def get(self, key: str) -> T:
        """Looks up a registered object by key.

        Args:
            key: The registered name.

        Returns:
            The registered object.

        Raises:
            KeyError: If ``key`` is not registered, listing valid keys.
        """
        if key not in self._items:
            raise KeyError(
                f'Unknown {self._name} {key!r}. '
                f'Registered: {sorted(self._items)}')
        return self._items[key]

    def keys(self) -> list:
        """Returns the sorted list of registered keys."""
        return sorted(self._items)

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)
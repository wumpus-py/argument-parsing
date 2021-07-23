from __future__ import annotations

from abc import ABC
from enums import Enum
from inspect import isclass
from itertools import chain

from typing import Literal, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Callable, Dict, List, Iterable, Tuple, TypeVar, overload

    A = TypeVar('A')
    L = TypeVar('L')
    ConverterT = Union['Converter', type, Callable[[None, str], A]]


__all__ = (
    'Converter',
    'Argument',
    'converter'
)


_NoneType: Type[None] = type(None)


class _NULL:
    ...


class ConversionError(Exception):
    ...  # Placeholder


class ConsumeType(Enum):
    """|enum|

    An enumeration of argument consumption types.

    Attributes
    ----------
    default
        The default and "normal" consumption type.
    consume_rest
        Consumes the string, including quotes, until the end.
    list
        Consumes in a similar fashion to the default consumption type,
        but will consume like this all the way until the end.
        If an error occurs, an error will be raised.
    tuple
        :attr:`.ConsumeType.list` except that the result is a tuple,
        rather than a list.
    greedy
        Like :attr:`.ConsumeType.list`, but it stops consuming
        when an argument fails to convert, rather than raising an error.
    """

    default: str      = 'default'
    consume_rest: str = 'consume_rest'
    list: str         = 'list'
    tuple: str        = 'tuple'
    greedy: str       = 'greedy'


class Converter(ABC):
    """A class that aids in making class-based converters."""

    __is_converter__: bool = True

    async def validate(self, ctx: ..., argument: str) -> bool:
        """|coro|

        The argument validation check to use.

        This will be called before convert and raise a :class:`.ValidationError`
        if it fails.

        This exists to encourage cleaner code.

        Parameters
        ----------
        ctx: Any
            Placeholder.
        argument: str
            The argument to validate.

        Returns
        -------
        bool
        """
        return True

    async def convert(self, ctx: ..., argument: str) -> A:
        """|coro|

        The core conversion of the argument.
        This must be implemented, or :class:`NotImplementedError` will be raised.

        Parameters
        ----------
        ctx: Any
            Placeholder.
        argument: str
            The argument to convert.

        Returns
        -------
        Any
        """
        raise NotImplementedError


class LiteralConverter(Converter):
    def __init__(self, /, *args: L) -> None:
        self._valid: Tuple[L, ...] = args

    async def convert(self, /, ctx: ..., argument: str) -> A:
        errors = []

        for possible in self._valid:
            p_type = type(possible)
            try:
                casted = await _convert_one(ctx, argument, p_type)
            except ConversionError as exc:
                errors.append(exc)
            else:
                if casted not in self._valid:
                    raise ConversionError(...)  # TODO: LiteralConversionError(ConversionError)

                return casted

        raise ConversionError(errors)  # TODO: LiteralConversionError(ConversionError)


def _sanitize_converter(converter: ConverterT, /) -> Tuple[Tuple[ConverterT], Optional[bool], Optional[ConsumeType]]:
    optional = consume = None

    if hasattr(converter, '__origin__') and hasattr(converter, '__args__'):
        origin = converter.__origin__
        if origin is Union:
            args = converter.__args__
            if _NoneType in args:
                # This is an optional type.
                optional = True
                return tuple(arg for arg in args if arg is not _NoneType), optional, consume

            return tuple(
                chain.from_iterable(_sanitize_converter(arg) for arg in args)
            ), optional, consume

        if origin is Literal:
            return LiteralConverter(*converter.__args__)

        if origin is List:
            consume = ConsumeType.list
            return (_sanitize_converter(converter.__args__[0]),), optional, consume

    if isclass(converter) and issubclass(converter, Converter):
        converter = converter()

    return converter, optional, consume


def _convert_bool(argument: str) -> A:
    if isinstance(argument, bool):
        return argument

    argument = argument.lower()
    if argument in {'true', 't', 'yes', 'y', 'on', 'enable', 'enabled', '1'}:
        return True
    if argument in {'false', 'f', 'no', 'n', 'off', 'disable', 'disabled', '0'}:
        return False

    raise ...  # TODO: BooleanConversionError(ConversionError)


async def _convert_one(ctx: ..., argument: str, converter: ConverterT) -> A:
    if converter is bool:
        return _convert_bool(converter)

    try:
        if getattr(converter, '__is_converter__', False):
            return await converter.convert(ctx, argument)

        return converter(argument)

    except Exception as exc:
        raise ConversionError(exc)


class Argument:
    """Represents a positional argument.

    Parameters
    ----------
    *converters
    """

    if TYPE_CHECKING:
        @overload
        def __init__(
            self,
            /,
            converter: ConverterT,
            *,
            name: str = None,
            signature: str = None,
            default: Any = _NULL,
            optional: bool = False,
            description: str = None,
            consume_type: Union[ConsumeType, str] = ConsumeType.default,
            quoted: bool = False,
            quotes: Dict[str, str] = None,
            **kwargs
        ) -> None:
            ...

        @overload
        def __init__(
            self,
            /,
            *converters: ConverterT,
            name: str = None,
            alias: str = None,
            signature: str = None,
            default: Any = _NULL,
            optional: bool = False,
            description: str = None,
            consume_type: Union[ConsumeType, str] = ConsumeType.default,
            quoted: bool = False,
            quotes: Dict[str, str] = None,
            **kwargs
        ) -> None:
            ...

    def __init__(
        self,
        /,
        *converters: ConverterT,
        name: str = None,
        signature: str = None,
        default: Any = _NULL,
        optional: bool = False,
        description: str = None,
        converter: ConverterT = None,
        consume_type: Union[ConsumeType, str] = ConsumeType.default,
        quoted: bool = False,
        quotes: Dict[str, str] = None,
        **kwargs
    ) -> None:
        actual_converters = str,

        if converters and converter is not None:
            raise ValueError('converter kwarg cannot be used when they are already passed as positional arguments.')

        if converters:
            actual_converters = converters
        elif converter:
            actual_converters = converter,


def converter(func: callable) -> Type[Converter]:
    """
    A decorator that helps convert a function into a converter.
    """

    class _Wrapper(Converter):
        async def convert(self, ctx: ..., argument: str) -> A:
            return await func(ctx, argument)  # TODO: maybe_coro this?

    return _Wrapper

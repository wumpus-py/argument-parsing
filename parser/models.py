from __future__ import annotations

from abc import ABC
from enums import Enum
from inspect import isclass
from itertools import chain

from typing import Generic, List, Literal, Union, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Callable, Dict, Iterable, Tuple, overload

    A = TypeVar('A')
    L = TypeVar('L')
    NT = TypeVar('NT', bound='ConverterT')
    ConverterT = Union['Converter', type, Callable[[None, str], A]]


G = TypeVar('G')
N = TypeVar('N')


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


class Quotes:
    """Builtin quote mappings. All attributes are instances of `dict[str, str]`

    Attributes
    ----------
    default
        The default quote mapping.
            `"` -> `"`
            `'` -> `'`
    extended
        An extended quote mapping that supports
        quotes from other languages/locales.

            `'"'`      -> `'"'`
            `'`        -> `'`
            `'\u2018'` -> `'\u2019'`
            `'\u201a'` -> `'\u201b'`
            `'\u201c'` -> `'\u201d'`
            `'\u201e'` -> `'\u201f'`
            `'\u2e42'` -> `'\u2e42'`
            `'\u300c'` -> `'\u300d'`
            `'\u300e'` -> `'\u300f'`
            `'\u301d'` -> `'\u301e'`
            `'\ufe41'` -> `'\ufe42'`
            `'\ufe43'` -> `'\ufe44'`
            `'\uff02'` -> `'\uff02'`
            `'\uff62'` -> `'\uff63'`
            `'\xab'`   -> `'\xbb'`
            `'\u2039'` -> `'\u203a'`
            `'\u300a'` -> `'\u300b'`
            `'\u3008'` -> `'\u3009'`
    """

    default = {
        '"': '"',
        "'": "'"
    }

    extended = {
        '"': '"',
        "'": "'",
        "\u2018": "\u2019",
        "\u201a": "\u201b",
        "\u201c": "\u201d",
        "\u201e": "\u201f",
        "\u2e42": "\u2e42",
        "\u300c": "\u300d",
        "\u300e": "\u300f",
        "\u301d": "\u301e",
        "\ufe41": "\ufe42",
        "\ufe43": "\ufe44",
        "\uff02": "\uff02",
        "\uff62": "\uff63",
        "\u2039": "\u203a",
        "\u300a": "\u300b",
        "\u3008": "\u3009",
        "\xab": "\xbb",
    }


class Greedy(Generic[G]):
    ...


class Not(Generic[N]):
    ...


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
    def __init__(self, /, *choices: L) -> None:
        self._valid: Tuple[L, ...] = choices

    def __repr__(self, /) -> None:
        return f'<{self.__class__.__name__} valid={self._valid!r}>'

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


class _Not(Converter):
    def __init__(self, /, *entities: NT) -> None:
        self._blacklist: Tuple[NT, ...] = entities

    def __repr__(self, /) -> None:
        return f'<{self.__class__.__name__} blacklist={self._blacklist!r}>'

    async def convert(self, /, ctx: ..., argument: str) -> A:
        for entity in self._blacklist:
            try:
                await _convert_one(ctx, argument, entity)
            except ConversionError:
                return argument
            else:
                raise ConversionError('argument is castable into {type}')  # TODO: NotConversionError


def _sanitize_converter(
    converter: ConverterT,
    /,
    optional: bool = None,
    consume: ConsumeType = None
) -> Tuple[Tuple[ConverterT], Optional[bool], Optional[ConsumeType]]:
    if hasattr(converter, '__origin__') and hasattr(converter, '__args__'):
        origin = converter.__origin__
        args = converter.__args__

        if origin is Union:
            if _NoneType in args:
                # This is an optional type.
                optional = True
                return tuple(arg for arg in args if arg is not _NoneType), optional, consume

            return tuple(
                chain.from_iterable(_sanitize_converter(arg)[0] for arg in args)
            ), optional, consume

        if origin is Literal:
            converter = LiteralConverter(*args)

        if origin is List:
            consume = ConsumeType.list
            return _sanitize_converter(args[0], optional, consume)

        if origin is Greedy:
            consume = ConsumeType.greedy

        if origin is Not:
            converter = _Not(*args)

    if isclass(converter) and issubclass(converter, Converter):
        converter = converter()

    return (converter,), optional, consume


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
            quoted: bool = None,
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
            quoted: bool = None,
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
        quoted: bool = None,
        quotes: Dict[str, str] = None,
        **kwargs
    ) -> None:
        actual_converters = str,

        if converters and converter is not None:
            raise ValueError('converter kwarg cannot be used when they are already passed as positional arguments.')

        if len(converters) == 1:
            converter = converters[0]
            converters = ()

        if converters or converter:
            if converters:
                actual_converters, optional_, consume = _sanitize_converter(converters[0])
                actual_converters += _sanitize_converter(converters[1:])[0]
            elif converter:
                actual_converters, optional_, consume = _sanitize_converter(converter)
            else:
                return  # to shut up my linter

            optional = optional if optional is not None else optional_
            consume_type = consume_type if consume_type is not None else consume

        self._param_key: Optional[str] = None
        self._param_positional: bool = False

        self.name: str = name
        self.optional: bool = optional
        self.description: str = description
        self.default: Any = default

        self.consume_type: ConsumeType = (
            consume_type if isinstance(consume_type, ConsumeType) else ConsumeType(consume_type)
        )

        self.quoted: bool = quoted if quoted is not None else consume_type is not ConsumeType.consume_rest
        self.quotes: Dict[str, str] = quotes if quotes is not None else Quotes.default

        self._signature: str = signature
        self._converters: Tuple[ConverterT, ...] = actual_converters

        self._kwargs: Dict[str, Any] = kwargs

    def __repr__(self, /) -> str:
        return f'<Argument name={self.name!r} optional={self.optional} consume_type={self.consume_type}>'

    @property
    def signature(self, /) -> str:
        """str: The signature of this argument."""

        if self._signature is not None:
            return self._signature

        start, end = '<>' if self.optional or self.default is not _NULL else '[]'

        suffix = '...' if self.consume_type in (
            ConsumeType.list, ConsumeType.tuple, ConsumeType.greedy
        ) else ''

        default = f'={self.default}' if self.default is not _NULL else ''
        return start + str(self.name) + default + suffix + end

    @signature.setter
    def signature(self, value: str, /) -> str:
        self._signature = value


def converter(func: callable) -> Type[Converter]:
    """
    A decorator that helps convert a function into a converter.
    """

    class _Wrapper(Converter):
        async def convert(self, ctx: ..., argument: str) -> A:
            return await func(ctx, argument)  # TODO: maybe_coro this?

    return _Wrapper

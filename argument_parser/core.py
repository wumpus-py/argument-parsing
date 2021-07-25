from __future__ import annotations

import inspect

from abc import ABC
from enum import Enum
from itertools import chain

from typing import Generic, List, Literal, Union, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Callable, Dict, Iterable, Tuple, Type, overload

    A = TypeVar('A')
    L = TypeVar('L')
    P = TypeVar('P', bound='Union[_Subparser, Parser]')

    ArgumentT = TypeVar('ArgumentT', bound='Argument')
    ConverterT = Union['Converter', type, Callable[[None, str], A]]
    ParserCallback = Callable[[Any, ConverterT, ...], Any]
    NT = TypeVar('NT', bound=ConverterT)


G = TypeVar('G')
N = TypeVar('N')


__all__ = (
    'Parser',
    'Converter',
    'ConversionError',
    'LiteralConverter',
    'Argument',
    'Greedy',
    'Not',
    'ConsumeType',
    'Quotes',
    'converter'
)


_NoneType: Type[None] = type(None)


class _NullType:
    def __bool__(self, /) -> bool:
        return False

    def __repr__(self, /) -> str:
        return 'NULL'

    __str__ = __repr__


_NULL = _NullType()


class ConversionError(Exception):
    ...  # Placeholder


class ArgumentParsingError(Exception):
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
                args = tuple(arg for arg in args if arg is not _NoneType)

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

    if inspect.isclass(converter) and issubclass(converter, Converter):
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


async def _convert(ctx: ..., argument: str, converters: Iterable[ConverterT]) -> A:
    errors = []
    for converter in converters:
        try:
            return await _convert_one(ctx, argument, converter)
        except ConversionError as exc:
            errors.append(exc)

    raise ConversionError(errors)  # TODO: ConversionFailure


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
            optional: bool = None,
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
            optional: bool = None,
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
        optional: bool = None,
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
        self._param_kwarg_only: bool = False

        self.name: str = name
        self.description: str = description
        self.default: Any = default

        self.consume_type: ConsumeType = (
            consume_type if isinstance(consume_type, ConsumeType) else ConsumeType(consume_type)
        )

        self.optional: bool = optional if optional is not None else False
        self.quoted: bool = quoted if quoted is not None else consume_type is not ConsumeType.consume_rest
        self.quotes: Dict[str, str] = quotes if quotes is not None else Quotes.default

        self._signature: str = signature
        self._converters: Tuple[ConverterT, ...] = actual_converters

        self._kwargs: Dict[str, Any] = kwargs

    def __repr__(self, /) -> str:
        return f'<Argument name={self.name!r} optional={self.optional} consume_type={self.consume_type}>'

    @property
    def converters(self, /) -> Tuple[ConverterT, ...]:
        """Tuple[Union[type, :class:`.Converter`], ...]: A tuple of this argument's converters."""
        return self._converters

    @property
    def signature(self, /) -> str:
        """str: The signature of this argument."""

        if self._signature is not None:
            return self._signature

        start, end = '[]' if self.optional or self.default is not _NULL else '<>'

        suffix = '...' if self.consume_type in (
            ConsumeType.list, ConsumeType.tuple, ConsumeType.greedy
        ) else ''

        default = f'={self.default}' if self.default is not _NULL else ''
        return start + str(self.name) + default + suffix + end

    @signature.setter
    def signature(self, value: str, /) -> str:
        self._signature = value

    @classmethod
    def _from_parameter(cls: Type[ArgumentT], param: inspect.Parameter, /) -> ArgumentT:
        def finalize(argument: ArgumentT) -> ArgumentT:
            if param.kind is param.KEYWORD_ONLY:
                argument._param_kwarg_only = True

            argument._param_key = param.name
            return argument

        kwargs = {'name': param.name}

        if param.annotation is not param.empty:
            if isinstance(param.annotation, cls):
                return finalize(param.annotation)

            kwargs['converter'] = param.annotation

        if param.default is not param.empty:
            kwargs['default'] = param.default

        if param.kind is param.KEYWORD_ONLY:
            kwargs['consume_type'] = ConsumeType.consume_rest

        elif param.kind is param.VAR_POSITIONAL:
            kwargs['consume_type'] = ConsumeType.tuple

        return finalize(cls(**kwargs))



class _StringReader:
    class EOF:
        ...

    def __init__(self, string: str, /, *, quotes: Dict[str, str] = None) -> None:
        self.quotes: Dict[str, str] = quotes or Quotes.default
        self.buffer: str = string
        self.index: int = -1

    def seek(self, index: int, /) -> str:
        self.index = index
        return self.current

    @property
    def current(self, /) -> str:
        try:
            return self.buffer[self.index]
        except IndexError:
            return self.EOF

    @property
    def eof(self, /) -> bool:
        return self.current is self.EOF

    def previous_character(self, /) -> str:
        return self.seek(self.index - 1)

    def next_character(self, /) -> str:
        return self.seek(self.index + 1)

    @property
    def rest(self, /) -> str:
        result = self.buffer[self.index:]
        self.index = len(self.buffer)  # Force an EOF
        return result

    @staticmethod
    def _is_whitespace(char: str, /) -> bool:
        if char is ...:
            return False
        return char.isspace()

    def skip_to_word(self, /) -> None:
        char = ...
        while self._is_whitespace(char):
            char = self.next_character()

    def next_word(self, *, skip_first: bool = True) -> str:
        char = ...
        buffer = ''

        if skip_first:
            self.skip_to_word()

        while not self._is_whitespace(char):
            char = self.next_character()
            if self.eof:
                return buffer

            buffer += char

        buffer = buffer[:-1]
        return buffer

    def next_quoted_word(self, *, skip_first: bool = True) -> str:
        if skip_first:
            self.skip_to_word()

        first_char = self.next_character()
        if first_char in self.quotes:
            end_quote = self.quotes[first_char]

            char = ...
            buffer = ''

            while char != end_quote or self.buffer[self.index - 1] == '\\':
                char = self.next_character()
                if self.eof:
                    return buffer

                buffer += char

            self.next_character()
            buffer = buffer[:-1]
            return buffer
        else:
            self.previous_character()
            return self.next_word(skip_first=False)


class _Subparser:
    """A class that parses one specific overload."""

    def __init__(self, arguments: List[Argument] = None, _flags: ... = None, /, *, callback: ParserCallback = None):
        self._arguments: List[Argument] = arguments or []
        self.callback: Optional[ParserCallback] = callback

    def add_argument(self, argument: Argument, /) -> None:
        self._arguments.append(argument)

    @property
    def signature(self, /) -> str:
        """str: The signature (or "usage string") for this parser."""
        return ' '.join(arg.signature for arg in self._arguments)

    @classmethod
    def from_function(cls: Type[P], func: ParserCallback, /) -> P:
        """Makes a :class:`.Parser` from a function."""

        params = list(inspect.signature(func).parameters.values())
        if len(params) < 1:
            raise TypeError('parser function must have at least one parameter (Context)')

        self = cls(callback=func)
        for param in params[1:]:
            self.add_argument(Argument._from_parameter(param))

        return self

    async def parse(self, text: str, /, ctx: ...) -> Tuple[List[Any], Dict[str, Any]]:
        # Return a tuple (args: list, kwargs: dict)
        # Execute as callback(ctx, *args, **kwargs)

        args = []
        kwargs = {}
        reader = _StringReader(text)

        def append_value(argument: Argument, value: Any, /) -> None:
            if argument._param_kwarg_only:
                kwargs[argument._param_key] = value
            else:
                args.append(value)

        i = 0

        for i, argument in enumerate(self._arguments, start=1):
            if reader.eof:
                i -= 1
                break

            start = reader.index

            if argument.consume_type not in (ConsumeType.consume_rest, ConsumeType.default):
                # Either list, tuple, or greedy
                result = []

                while not reader.eof:
                    word = reader.next_quoted_word() if argument.quoted else reader.next_word()
                    try:
                        word = await _convert(ctx, word, argument.converters)
                    except ConversionError as exc:
                        if argument.consume_type is not ConsumeType.greedy:
                            raise exc
                        break
                    else:
                        result.append(word)

                if argument.consume_type is ConsumeType.tuple:
                    result = tuple(result)

                append_value(argument, result)
                continue

            if argument.consume_type is ConsumeType.consume_rest:
                word = reader.rest.strip()
            else:
                word = reader.next_quoted_word() if argument.quoted else reader.next_word()

            try:
                word = await _convert(ctx, word, argument.converters)
            except ConversionError as exc:
                if argument.optional:
                    default = argument.default if argument.default is not _NULL else None
                    append_value(argument, default)
                    reader.seek(start)
                    continue

                raise exc
            else:
                append_value(argument, word)

        if i < len(self._arguments):
            for argument in self._arguments[i:]:
                if argument.default is not _NULL:
                    append_value(argument, argument.default)
                elif argument.optional:
                    append_value(argument, None)
                else:
                    raise ArgumentParsingError(f'missing required argument {argument.name!r}')
                    # TODO: MissingArgumentError (or whatever)

        return args, kwargs


class Parser:
    """The main class that parses arguments."""

    def __init__(self, /, *, overloads: List[_Subparser] = None) -> None:
        self._overloads: List[_Subparser] = overloads or []

    @property
    def main_parser(self, /) -> _Subparser:
        if not len(self._overloads):
            self._overloads.append(_Subparser())

        return self._overloads[0]

    def add_argument(self, argument: Argument, /) -> None:
        self.main_parser.add_argument(argument)

    @property
    def signature(self, /) -> str:
        """str: The signature (or "usage string") for this parser."""
        return self.main_parser.signature

    def overload(self, func: ParserCallback, /) -> _Subparser:
        result = _Subparser.from_function(func)
        self._overloads.append(result)
        return result

    @classmethod
    def from_function(cls: Type[P], func: ParserCallback, /) -> P:
        parser = _Subparser.from_function(func)
        return cls(overloads=[parser])

    async def parse(self, text: str, /, ctx: ...) -> Tuple[ParserCallback, List[Any], Dict[str, Any]]:
        errors = []
        for overload in self._overloads:
            try:
                return overload.callback, *await overload.parse(text, ctx=ctx)
            except Exception as exc:
                errors.append(exc)

        raise ArgumentParsingError(errors)  # TODO: ArgumentParsingError

    async def execute(self, /, ctx: ..., content: str) -> None:
        callback, args, kwargs = await self.parse(content, ctx=ctx)
        await callback(ctx, *args, **kwargs)


def converter(func: callable) -> Type[Converter]:
    """
    A decorator that helps convert a function into a converter.
    """

    class _Wrapper(Converter):
        async def convert(self, ctx: ..., argument: str) -> A:
            return await func(ctx, argument)  # TODO: maybe_coro this?

    return _Wrapper

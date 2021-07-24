# argument-parsing
Prototyping the argument parser for wumpus.py's command plugin.

This does not include any implementation of Discord, this is just the bare argument parser.

## Features
- Implicit conversion and casting
- Consume rest behaviour and custom consumption
- Command/argument "overloading"
- Argument validation
- Flags
- Wide range of customization

## Usage
```py
from argument_parser import Argument, parse

@parse()
def add(
    x: Argument(int),
    y: Argument(int, default=5),
    comment: Argument(consume_rest=True, default='')
):
    print(x + y, comment)
    
add.parse('4 5 this is a comment')
```

### Implicit annotations (No use of `Argument`)
```py
from argument_parser import parse

@parse()
def add(
    x: int,
    y: int = 5,
    *,
    comment: str = ''
):
    print(x + y)
    
add.parse('4 5 this is a comment')
```

### Argument validation
```py
from argument_parser import Argument

@parse()
def add(
    # Builtin validations
    x: Argument(int, min_value=0, max_value=100),
    
    # Custom validations
    y: Argument(int, default=5, check=lambda arg: 0 <= arg <= 100)
):
    ...
```

### Flags
```py
from argument_parser import Argument, Flag, ShorthandFlagAlias as Short

@parse()
def add(
    x: Argument(int),
    y: Argument(int),
    comment: Flag(alias=Short('c', prefix='-'), prefix='--')
):
    ...
    
add.parse('5 4 --comment This is a comment')
```

### Overloading
```py
from argument_parser import Argument

@parse()
def add(
    x: Argument(int)
):
    print(f'Callback 1, {x}')
    
@add.overload
def add_overload(
    x: Argument(str)
):
    print(f'Callback 2, {x}')
    
add.parse('4')
add.parse('test')
```

def parse_var(s):
    """
    Parse a key, value pair, separated by '='
    That's the reverse of ShellArgs.

    On the command line (argparse) a declaration will typically look like:
        foo=hello
    or
        foo="hello world"
    """
    items = s.split('=')
    key = items[0].strip() # we remove blanks around keys, as is logical
    if len(items) > 1:
        # rejoin the rest:
        value = '='.join(items[1:])
    return (key, value)


def parse_vars(items):
    """
    Parse a series of key-value pairs and return a dictionary
    """
    d = {}
    
    if items:
        for item in items:
            key, value = parse_var(item)
            d[key] = value
    return d

def parse_slice(st):
    """
    Parse a string into a slice object.
    The string should be in the format "start:stop:step".
    """
    if st == "":
        return slice(None)
    if ":" not in st:
        return slice(int(st), None, None)
    parts = st.split(":")
    if len(parts) == 1:
        return slice(int(parts[0]), None, None)
    elif len(parts) == 2:
        return slice(int(parts[0]), int(parts[1]), None)
    elif len(parts) == 3:
        return slice(int(parts[0]), int(parts[1]), int(parts[2]))
    else:
        raise ValueError(f"Invalid slice format: {st}")
    
    
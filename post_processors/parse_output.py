def parse_cpp_output(output):
    """Parses the output of the C++ program into a dictionary."""
    lines = output.strip().split("\n")
    parsed_data = {}
    
    for line in lines:
        key, value = line.split("=")
        parsed_data[key] = convert_value(value)
    
    return parsed_data

def convert_value(value):
    """Converts the string value to its appropriate Python type."""
    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False

    return value

if __name__ == "__main__":
    # Assuming the C++ program's output is stored in the variable 'cpp_output'
    cpp_output = """
    K=6
    L=12
    Lnn=2
    ... [and so on]
    """
    parsed_data = parse_cpp_output(cpp_output)
    for key, value in parsed_data.items():
        print(f"{key}: {value}")

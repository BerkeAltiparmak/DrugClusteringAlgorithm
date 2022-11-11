def check_if_number(value: str) -> bool:
    if value[0] not in ["+", "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
        return False
    for c in value[1: len(value)]:
        if c not in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            return False

    return True


def check_if_operation(value: str) -> bool:
    if value == "+" or value == "-":
        return True
    else:
        return False


def check_expression(expression: str) -> str:
    first_num = None #int
    operation1 = None  # string
    second_num = None #int
    operation2 = None  # string
    third_num = None # int

    value = ""  #string
    lhs = 0
    expression = expression + " "
    for c in expression: # iterating through each character in expression
        value = value + c

        if first_num is None:
            if not check_if_number(value):
                value = value[0: len(value) - 1]
                if check_if_number(value):
                    first_num = int(value)
                    value = c
                else:
                    return "invalid expression"
        elif operation1 is None:
            if not check_if_operation(value):
                value = value[0: len(value) - 1]
                if check_if_operation(value):
                    operation1 = value
                    value = c
                else:
                    return "invalid expression"
        elif second_num is None:
            if not check_if_number(value):
                value = value[0: len(value) - 1]
                if check_if_number(value):
                    second_num = int(value)
                    value = c
                else:
                    return "invalid expression"
        elif operation2 is None:
            if value != "=":
                value = value[0: len(value) - 1]
                if value == "=":
                    operation2 = value
                    value = c
                else:
                    return "invalid expression"
        elif third_num is None:
            if not check_if_number(value):
                value = value[0: len(value) - 1]
                if check_if_number(value):
                    third_num = int(value)
                    value = c
                else:
                    return "invalid expression"
    if operation1 == "+":
        lhs = first_num + second_num
    elif operation1 == "-":
        lhs = first_num - second_num
    rhs = third_num

    if lhs == rhs:
        return "correct answer"

    return "incorrect answer"




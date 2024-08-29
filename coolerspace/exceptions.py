class OperandShapeMismatchError(Exception):
    """
    Thrown when the shape of operands are not valid for a given operation
    """
    pass


class InvalidConstantShapeError(Exception):
    """
    Thrown when an invalid shape is used to initialize a constant
    """
    pass


class ColorTypeMismatchError(Exception):
    """
    Thrown when an operation occurs between two color types of different types
    """
    pass


class OperationNotFoundError(Exception):
    """
    Thrown when a requested operation is not found
    """
    pass


class NonConcreteTypeError(Exception):
    """
    Thrown when a user attempts to create an output from a non-concrete type
    """
    pass


class InvalidTransformationError(Exception):
    """
    Thrown when the user-specified transformation is impossible
    """
    pass
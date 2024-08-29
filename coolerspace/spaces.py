from __future__ import annotations

import numpy as np
import networkx as nx
import onnx

from typing import Type, Optional, Callable
from abc import ABC
from enum import Enum

from .construction import OnnxConstructor
from .transformation import TransformationFunctions, color_transformation_generator
from .operation import ShapeValidationFunctions, OperationFunctions, OperationHelpers
from .exceptions import *


# region ParentType
class ParentType(ABC):
    def __init__(
            self,
            value: Optional[ParentType | np.ndarray | list] = None,
            # The below are only set if value is not specified
            onnx_name: Optional[str] = None,
            onnx_shape: Optional[list[int]] = None,
            color_shape: Optional[list[int]] = None,
            operand_list: Optional[list[ParentType]] = None,
            color_operator: Optional[OperationManager.ConcreteOperation] = None
    ):
        # Case 1: Constant as numpy array
        if isinstance(value, np.ndarray):
            self._init_constant(value)
            return

        # Case 2: Constant as sequence
        elif isinstance(value, list):
            self._init_constant(np.asarray(value))
            return

        # Case 2: Casting
        elif isinstance(value, ParentType):
            self._init_transform(value)
            return

        # Case 3: Value is not valid
        elif value is not None:
            assert False

        # Case 4: Developer initialization
        self._onnx_name = onnx_name
        """
        The name of the output of the onnx node
        """
        self._onnx_shape = onnx_shape
        """
        The shape of the node represented in onnx. Does not omit last dimension.
        A 1920x1080 RGB image will have an onnx shape of 1920x1080x3.
        """
        self._color_shape = color_shape
        """
        The shape of the color object. Omits the last dimension.
        For example, a 1920x1080 image will have a color shape of 1920x1080 instead of 1920x1080x3
        """
        self._operand_list = operand_list
        """
        An ordered list of operands.
        """
        self._color_operator = color_operator
        """
        The corresponding color operator of the parent type, if applicable.
        NOT the onnx operator.
        """

    def _init_constant(self, value: np.ndarray):
        """
        Initializes a constant colorspace node.
        """
        # Validate value shape. Last dimension must be equal to channel count if channel count > 1
        expected_onnx_shape = list(value.shape)
        # If constant shape has an invalid number of channels, throw error
        if not self._validate_onnx_shape(expected_onnx_shape):
            raise InvalidConstantShapeError(
                "The constant provided for initializing the type {} was invalid. Expected {} channels.".format(
                    self.__class__,
                    self.channel_count
                )
            )

        # Create constant node
        onnx_output_name = OnnxConstructor.create_constant(value)

        # Initialize
        self._onnx_name = onnx_output_name
        self._onnx_shape = expected_onnx_shape
        self._color_shape = self._onnx_shape_to_color_shape(expected_onnx_shape)

    def _init_transform(self, original: ParentType):
        """
        Initializes a transformation colorspace node.
        """
        # Special case: Casting from a pigment (should extract reflectance, absorption, or scattering spectrum)
        if isinstance(original, Pigment):
            target_class = self.__class__
            if target_class not in [ScatteringSpectrum, ReflectanceSpectrum, AbsorptionSpectrum]:
                raise InvalidTransformationError("Can only transform a Pigment object to a scattering, reflectance, or absorption spectrum.")

            if target_class == ScatteringSpectrum:
                onnx_name = original._scattering_onnx_name
            elif target_class == ReflectanceSpectrum:
                onnx_name = original._onnx_name
            elif target_class == AbsorptionSpectrum:
                onnx_name = original._absorption_onnx_name

            self._onnx_name = onnx_name
            self._onnx_shape = original._onnx_shape
            self._color_shape = original._color_shape

            return


        # Check if a transformation path exists
        transformation_path_exists = TransformationManager.check_transformation_validity(
            origin_class=original.__class__,
            destination_class=self.__class__
        )

        if not transformation_path_exists:
            raise InvalidTransformationError("The casting from {} to {} is impossible.".format(original.__class__.__name__, self.__class__.__name__))

        # Create transformation nodes
        onnx_output_name = TransformationManager.construct_transformation_onnx_nodes(
            origin_object=original,
            destination_class=self.__class__
        )

        # Initialize object
        self._onnx_name = onnx_output_name
        self._onnx_shape = OnnxConstructor.get_shape(self._onnx_name)
        self._color_shape = self._onnx_shape_to_color_shape(self._onnx_shape)

    channel_count: int = None
    """
    Defines the number of channels of the type.
    For example, RGB has a channel count of 3.
    CMYK has a channel count of 4.
    """
    _matrixable: bool = True
    """
    Indicates if the type is able to be cast to and from matrix form
    """

    @classmethod
    def _color_shape_to_onnx_shape(cls, color_shape: list[int]) -> list[int]:
        """
        Converts color shape to onnx shape
        """
        # Make sure we dupe to avoid mutating original
        if isinstance(color_shape, int):
            color_shape = [color_shape]
        onnx_shape = color_shape.copy()

        # Only append if the channel count is more than 1
        if cls.channel_count > 1:
            onnx_shape.append(cls.channel_count)

        return onnx_shape

    @classmethod
    def _onnx_shape_to_color_shape(cls, onnx_shape: list[int] | int) -> list[int]:
        """
        Converts onnx shape to color shape
        """
        # Make sure we dupe to avoid mutating original
        if isinstance(onnx_shape, int):
            onnx_shape = [onnx_shape]
        color_shape = onnx_shape.copy()

        # If the channel count is 1 or less, color shape is onnx shape
        if cls.channel_count <= 1:
            return color_shape

        # Otherwise, we check to ensure that the final dimension is correct then remove it
        assert color_shape[-1] == cls.channel_count
        return color_shape[0:-1]

    @classmethod
    def _validate_onnx_shape(cls, onnx_shape: list[int]) -> bool:
        """
        Checks to ensure that the final dimension of the onnx shape is equal to the channel count if the channel count is greater than 1.
        :returns: True if onnx shape is valid. False otherwise.
        """
        return cls.channel_count <= 1 or onnx_shape[-1] == cls.channel_count

    # region Operation catchers
    def __add__(self, other: ParentType | np.ndarray | int | float) -> ParentType:
        # Ensure that other is of a valid type
        if not (isinstance(other, ParentType) or isinstance(other, np.ndarray) or
                isinstance(other, int) or isinstance(other, float)):
            return NotImplemented

        # Cast into matrix type if applicable
        other = self._cast_into_matrix_if_applicable(other)

        # Construct
        return OperationManager.construct_operation(
            OperationManager.SemanticOperationEnum.ADD,
            self,
            other
        )

    def __radd__(self, other: ParentType | np.ndarray | int | float) -> ParentType:
        # Ensure that other is of a valid type
        if not (isinstance(other, ParentType) or isinstance(other, np.ndarray) or
                isinstance(other, int) or isinstance(other, float)):
            return NotImplemented

        # Cast into matrix type if applicable
        other = self._cast_into_matrix_if_applicable(other)

        # Construct
        return OperationManager.construct_operation(
            OperationManager.SemanticOperationEnum.ADD,
            other,
            self
        )

    def __mul__(self, other: ParentType | np.ndarray | int | float) -> ParentType:
        # Ensure that other is of a valid type
        if not (isinstance(other, ParentType) or isinstance(other, np.ndarray) or
                isinstance(other, int) or isinstance(other, float)):
            return NotImplemented

        # Cast into matrix type if applicable
        other = self._cast_into_matrix_if_applicable(other)

        # Construct
        return OperationManager.construct_operation(
            OperationManager.SemanticOperationEnum.MUL,
            self,
            other
        )

    def __rmul__(self, other: ParentType | np.ndarray | int | float) -> ParentType:
        # Ensure that other is of a valid type
        if not (isinstance(other, ParentType) or isinstance(other, np.ndarray) or
                isinstance(other, int) or isinstance(other, float)):
            return NotImplemented

        # Cast into matrix type if applicable
        other = self._cast_into_matrix_if_applicable(other)

        # Construct
        return OperationManager.construct_operation(
            OperationManager.SemanticOperationEnum.MUL,
            other,
            self
        )

    def __truediv__(self, other: ParentType | np.ndarray | int | float) -> ParentType:
        # Ensure that other is of a valid type
        if not (isinstance(other, ParentType) or isinstance(other, np.ndarray) or
                isinstance(other, int) or isinstance(other, float)):
            return NotImplemented

        # Cast into matrix type if applicable
        other = self._cast_into_matrix_if_applicable(other)

        # Construct
        return OperationManager.construct_operation(
            OperationManager.SemanticOperationEnum.DIV,
            self,
            other
        )

    def __rtruediv__(self, other: ParentType | np.ndarray | int | float) -> ParentType:
        # Ensure that other is of a valid type
        if not (isinstance(other, ParentType) or isinstance(other, np.ndarray) or
                isinstance(other, int) or isinstance(other, float)):
            return NotImplemented

        # Cast into matrix type if applicable
        other = self._cast_into_matrix_if_applicable(other)

        # Construct
        return OperationManager.construct_operation(
            OperationManager.SemanticOperationEnum.DIV,
            other,
            self
        )

    def __sub__(self, other: ParentType | np.ndarray | int | float) -> ParentType:
        # Ensure that other is of a valid type
        if not (isinstance(other, ParentType) or isinstance(other, np.ndarray) or
                isinstance(other, int) or isinstance(other, float)):
            return NotImplemented

        # Cast into matrix type if applicable
        other = self._cast_into_matrix_if_applicable(other)

        # Construct
        return OperationManager.construct_operation(
            OperationManager.SemanticOperationEnum.SUB,
            self,
            other
        )

    def __rsub__(self, other: ParentType | np.ndarray | int | float) -> ParentType:
        # Ensure that other is of a valid type
        if not (isinstance(other, ParentType) or isinstance(other, np.ndarray) or
                isinstance(other, int) or isinstance(other, float)):
            return NotImplemented

        # Cast into matrix type if applicable
        other = self._cast_into_matrix_if_applicable(other)

        # Construct
        return OperationManager.construct_operation(
            OperationManager.SemanticOperationEnum.SUB,
            other,
            self
        )

    @staticmethod
    def _cast_into_matrix_if_applicable(other: ParentType | np.ndarray | int | float):
        # If other input is int or float
        if isinstance(other, int) or isinstance(other, float):
            # Cast to numpy array
            other = np.asarray(other)

        # If still a numpy array, make it into a matrix object
        if isinstance(other, np.ndarray):
            other = Matrix(other)

        return other
    # endregion

    def _get_channel_by_number(self, channel: int):
        """
        Hidden function to get a specific channel of an object
        """
        return ChannelFetcher(self, channel)

    def _get_channel_cutter(self, channel: int) -> Matrix:
        """
        The channel cutter removes the target channel from the original array
        """
        cutter_list = [
            [1 if x == y and x != channel else 0 for x in range(self.channel_count)]
            for y in range(self.channel_count)
        ]

        return Matrix(cutter_list)

    def _get_channel_projector(self, channel: int) -> Matrix:
        """
        The channel projector projects the channel back to the original shape
        """
        projector_list = [[1 if i == channel else 0 for i in range(self.channel_count)]]

        return Matrix(projector_list)

    def _set_channel_by_number(self, channel: int, value: Matrix):
        """
        Hidden function to override onnx name
        """
        # Ensure that we are assigning a matrix
        if type(value) != Matrix:
            raise TypeError("Can only assign a matrix type to a channel.")

        # Cut original
        self_matrix = Matrix(self)
        self_cut = matmul(self_matrix, self._get_channel_cutter(channel))

        # Project new value
        value_projected = matmul(value, self._get_channel_projector(channel))

        # Add projected to original
        new_value = self_cut + value_projected

        # The new and old shapes should be the same
        assert new_value._onnx_shape == self._onnx_shape

        # If valid, assign new name
        self._onnx_name = new_value._onnx_name
# endregion


# region Channel fetching
class ChannelFetcher:
    """
    Wrapper to allow for operation overloading on channels
    """
    def __init__(self, original: ParentType, channel: int):
        # Validate channel count
        if channel < 0 or channel >= original.channel_count:
            assert False

        self.original: ParentType = original
        """
        Pointer to original object
        """
        self.channel: int = channel
        """
        Specific channel that the ChannelFetcher object represents
        """
        self.cache: Optional[Matrix] = None
        """
        Cache of computed channel. Computed on demand.
        """

    @property
    def matrix(self) -> Matrix:
        if self.cache is None:
            self.cache = matmul(Matrix(self.original), self._get_channel_extractor())

        return self.cache

    # region Helper matrices
    def _get_channel_extractor(self) -> Matrix:
        """
        Fetches channel extractor
        """
        extractor_list = [[1] if i == self.channel else [0] for i in range(self.original.channel_count)]
        return Matrix(extractor_list)

    def _get_channel_cutter(self) -> Matrix:
        """
        The channel cutter removes the target channel from the original array
        """
        cutter_list = [
            [1 if x == y and x != self.channel else 0 for x in range(self.original.channel_count)]
            for y in range(self.original.channel_count)
        ]

        return Matrix(cutter_list)

    def _get_channel_projector(self) -> Matrix:
        """
        The channel projector projects the channel back to the original shape
        """
        projector_list = [[1 if i == self.channel else 0 for i in range(self.original.channel_count)]]

        return Matrix(projector_list)
    # endregion

    # region Arithmetic operator overloads
    def __add__(self, other: ParentType | ChannelFetcher | int | float | np.ndarray):
        # Ensure that type is within bounds
        if not (isinstance(other, ParentType) or isinstance(other, ChannelFetcher) or
                isinstance(other, int) or isinstance(other, float) or isinstance(other, np.ndarray)):
            return NotImplemented

        return self.matrix + other

    def __radd__(self, other: ParentType | ChannelFetcher | int | float | np.ndarray):
        # Ensure that type is within bounds
        if not (isinstance(other, ParentType) or isinstance(other, ChannelFetcher) or
                isinstance(other, int) or isinstance(other, float) or isinstance(other, np.ndarray)):
            return NotImplemented

        return other + self.matrix

    def __sub__(self, other: ParentType | ChannelFetcher | int | float | np.ndarray):
        # Ensure that type is within bounds
        if not (isinstance(other, ParentType) or isinstance(other, ChannelFetcher) or
                isinstance(other, int) or isinstance(other, float) or isinstance(other, np.ndarray)):
            return NotImplemented

        return self.matrix - other

    def __rsub__(self, other: ParentType | ChannelFetcher | int | float | np.ndarray):
        # Ensure that type is within bounds
        if not (isinstance(other, ParentType) or isinstance(other, ChannelFetcher) or
                isinstance(other, int) or isinstance(other, float) or isinstance(other, np.ndarray)):
            return NotImplemented

        return other - self.matrix

    def __mul__(self, other: ParentType | ChannelFetcher | int | float | np.ndarray):
        # Ensure that type is within bounds
        if not (isinstance(other, ParentType) or isinstance(other, ChannelFetcher) or
                isinstance(other, int) or isinstance(other, float) or isinstance(other, np.ndarray)):
            return NotImplemented

        return self.matrix * other

    def __rmul__(self, other: ParentType | ChannelFetcher | int | float | np.ndarray):
        # Ensure that type is within bounds
        if not (isinstance(other, ParentType) or isinstance(other, ChannelFetcher) or
                isinstance(other, int) or isinstance(other, float) or isinstance(other, np.ndarray)):
            return NotImplemented

        return other * self.matrix

    def __truediv__(self, other: ParentType | ChannelFetcher | int | float | np.ndarray):
        # Ensure that type is within bounds
        if not (isinstance(other, ParentType) or isinstance(other, ChannelFetcher) or
                isinstance(other, int) or isinstance(other, float) or isinstance(other, np.ndarray)):
            return NotImplemented

        return self.matrix / other

    def __rtruediv__(self, other: ParentType | ChannelFetcher | int | float | np.ndarray):
        # Ensure that type is within bounds
        if not (isinstance(other, ParentType) or isinstance(other, ChannelFetcher) or
                isinstance(other, int) or isinstance(other, float) or isinstance(other, np.ndarray)):
            return NotImplemented

        return other / self.matrix
    # endregion
# endregion


# region Color types
class ColorType(ParentType, ABC):
    # Color type specific properties
    bit_depth: Optional[int] = None
    """
    Bit depth of the color space
    """
    gamma: Optional[float] = None
    """
    Gamma of the color space
    """


class XYZ(ColorType):
    bit_depth = None
    gamma = None
    channel_count = 3

    # Channels
    @property
    def X(self):
        return self._get_channel_by_number(0)

    @X.setter
    def X(self, value):
        self._set_channel_by_number(0, value)

    @property
    def Y(self):
        return self._get_channel_by_number(1)

    @Y.setter
    def Y(self, value):
        self._set_channel_by_number(1, value)

    @property
    def Z(self):
        return self._get_channel_by_number(2)

    @Z.setter
    def Z(self, value):
        self._set_channel_by_number(2, value)


class sRGB(ColorType):
    bit_depth = 8
    gamma = 2.2
    channel_count = 3

    # Channels
    @property
    def R(self):
        return self._get_channel_by_number(0)

    @R.setter
    def R(self, value):
        self._set_channel_by_number(0, value)

    @property
    def G(self):
        return self._get_channel_by_number(1)

    @G.setter
    def G(self, value):
        self._set_channel_by_number(1, value)

    @property
    def B(self):
        return self._get_channel_by_number(2)

    @B.setter
    def B(self, value):
        self._set_channel_by_number(2, value)


class opRGB(ColorType):
    bit_depth = 8
    gamma = 563/256
    channel_count = 3

    # Channels
    @property
    def R(self):
        return self._get_channel_by_number(0)

    @R.setter
    def R(self, value):
        self._set_channel_by_number(0, value)

    @property
    def G(self):
        return self._get_channel_by_number(1)

    @G.setter
    def G(self, value):
        self._set_channel_by_number(1, value)

    @property
    def B(self):
        return self._get_channel_by_number(2)

    @B.setter
    def B(self, value):
        self._set_channel_by_number(2, value)


class LMS(ColorType):
    """
    Based on Stockman & Sharpe 2000 2 degree cone fundamentals
    """
    bit_depth = None
    gamma = None
    channel_count = 3

    # Channels
    @property
    def L(self):
        return self._get_channel_by_number(0)

    @L.setter
    def L(self, value):
        self._set_channel_by_number(0, value)

    @property
    def M(self):
        return self._get_channel_by_number(1)

    @M.setter
    def M(self, value):
        self._set_channel_by_number(1, value)

    @property
    def S(self):
        return self._get_channel_by_number(2)

    @S.setter
    def S(self, value):
        self._set_channel_by_number(2, value)
# endregion


# region Spectra types
class SpectrumType(ParentType, ABC):
    """
    Spectrum classes store data from 390 to 830 nm at buckets of size 5
    """
    channel_count = 89


class AbsorptionSpectrum(SpectrumType):
    pass


class ReflectanceSpectrum(SpectrumType):
    pass


class ScatteringSpectrum(SpectrumType):
    pass


class LightSpectrum(SpectrumType):
    pass
# endregion


# region Perceptual color spaces
class LAB(ParentType):
    """
    CIELAB color space. Standard illuminant used is D65.
    """
    channel_count = 3

    # Channels
    @property
    def L(self):
        return self._get_channel_by_number(0)

    @property
    def A(self):
        return self._get_channel_by_number(1)

    @property
    def B(self):
        return self._get_channel_by_number(2)


class HSV(ParentType):
    """
    HSV color space.
    """
    channel_count = 3

    # Channels
    @property
    def H(self):
        return self._get_channel_by_number(0)

    @property
    def S(self):
        return self._get_channel_by_number(1)

    @property
    def V(self):
        return self._get_channel_by_number(2)
# endregion


# region Pigment hell
class Pigment(ParentType):
    _matrixable = False

    def __init__(self, value1: Optional[AbsorptionSpectrum, ScatteringSpectrum] = None,
                 value2: Optional[AbsorptionSpectrum, ScatteringSpectrum] = None):
        # We do not allow initialization of pigment type through constants!
        # There should be at least 1 absorption spectrum and 1 scattering spectrum provided
        if frozenset([value1.__class__, value2.__class__]) != frozenset([AbsorptionSpectrum, ScatteringSpectrum]):
            raise TypeError("Input classes to Pigment constructor must be of AbsorptionSpectrum and ScatteringSpectrum types")

        # The dimensions of the two values should match
        if value1._onnx_shape != value2._onnx_shape:
            raise TypeError("The dimensions of the two input spectra must match.")

        # Assign values to scattering/absorption
        scattering = value1 if value1.__class__ == ScatteringSpectrum else value2
        absorption = value1 if value1.__class__ == AbsorptionSpectrum else value2

        # Assign members
        self._reflectance_onnx_name = None
        self._scattering_onnx_name = scattering._onnx_name
        self._absorption_onnx_name = absorption._onnx_name
        self._onnx_shape = scattering._onnx_shape
        self._color_shape = scattering._color_shape

    @property
    def _onnx_name(self):
        if self._reflectance_onnx_name is not None:
            return self._reflectance_onnx_name
        else:
            self._reflectance_onnx_name = OperationFunctions.calculate_reflectance_from_absorption_scattering(
                self._absorption_onnx_name,
                self._scattering_onnx_name
            )
            return self._reflectance_onnx_name

# endregion


# region Misc types
class Chromaticity(ParentType):
    channel_count = 2


class Matrix(ParentType):
    channel_count = 1
# endregion Misc types


# region User-exposed I/O functions
def create_input(name: str, shape: list[int], colorspace: Type[ParentType]) -> ParentType:
    """
    Creates an input to the program.
    :param name: Name of the input
    :param shape: Color shape of the input
    :param colorspace: Colorspace of the input
    :returns: Generated input object
    """
    # The input is a color shape. Translate it to onnx shape.
    onnx_shape = colorspace._color_shape_to_onnx_shape(shape)

    # Generate an input
    input_creation_successful = OnnxConstructor.create_input(
        name=name,
        shape=onnx_shape
    )

    assert input_creation_successful

    # Create new object
    return colorspace(onnx_name=name, onnx_shape=onnx_shape, color_shape=shape)


create_output_type_blacklist: list[Type[ParentType]] = [
    Pigment
]
"""
Types that cannot be directly output
"""


def create_output(output_object: ParentType, data_type = onnx.TensorProto.DOUBLE):
    # We should not output types that are in the output type blacklist
    if output_object.__class__ in create_output_type_blacklist:
        raise NonConcreteTypeError("Cannot create output of object of type {}".format(output_object.__class__.__name__))

    # Create output
    OnnxConstructor.create_output(
        name=output_object._onnx_name,
        shape=output_object._onnx_shape,
        data_type=data_type
    )


def compile(output_path: str):
    """
    Saves generated onnx graph to file
    """
    model = OnnxConstructor.generate_model()
    onnx.save_model(model, output_path)


# endregion


# region User-exposed operation functions
def matmul(left_operand: ParentType | int | float | np.ndarray , right_operand: ParentType | int | float | np.ndarray):
    """
    Performs matrix multiplication between the two operands
    """
    # Ensure left and right operands match
    left_operand = ParentType._cast_into_matrix_if_applicable(left_operand)
    right_operand = ParentType._cast_into_matrix_if_applicable(right_operand)

    return OperationManager.construct_operation(
        semantic_op=OperationManager.SemanticOperationEnum.MATMUL,
        left_operand=left_operand,
        right_operand=right_operand
    )


def mix(*args):
    """
    Mixes n pigment objects given corresponding concentrations.
    Uses K-M model.
    """
    # Should have even # of arguments, and should have at least 4 arguments
    if len(args) % 2 != 0:
        raise TypeError("mix() must have an even number of arguments.")
    if len(args) < 4:
        raise TypeError("mix() must have at least two sets of pigments.")

    # Even arguments are weights, odd arguments are pigments
    weights: list = []
    pigments: list[Pigment] = []
    for i in range(len(args)):
        if i % 2 == 0:
            if isinstance(args[i], Matrix):
                assert ShapeValidationFunctions.broadcastable_match(args[i+1], args[i])
                weights.append(args[i])
            elif isinstance(args[i], float) or isinstance(args[i], int):
                weights.append(Matrix(args[i]))
            else:
                assert False
        else:
            assert isinstance(args[i], Pigment)
            pigments.append(args[i])

    absorption, scattering = OperationHelpers.ks_weighted(
        weights=weights,
        absorption_spectra=[pigment._absorption_onnx_name for pigment in pigments],
        scattering_spectra=[pigment._scattering_onnx_name for pigment in pigments]
    )

    # Wrap in proper colorspace objects
    absorption_shape = OnnxConstructor.get_shape(absorption)
    absorption_object = AbsorptionSpectrum(
        onnx_name=absorption,
        onnx_shape=absorption_shape,
        color_shape=AbsorptionSpectrum._onnx_shape_to_color_shape(absorption_shape),
    )

    scattering_shape = OnnxConstructor.get_shape(scattering)
    scattering_object = ScatteringSpectrum(
        onnx_name=scattering,
        onnx_shape=scattering_shape,
        color_shape=ScatteringSpectrum._onnx_shape_to_color_shape(scattering_shape)
    )

    return Pigment(absorption_object, scattering_object)

# endregion


# region Transformation logic
class TransformationManager:
    """
    Singleton class to check transformation validity and to translate transformations to onnx nodes
    """
    __node_list: list[Type[ParentType]] = [
        XYZ,
        sRGB,
        Chromaticity,
        ScatteringSpectrum,
        LightSpectrum,
        ReflectanceSpectrum,
        AbsorptionSpectrum,
        LAB,
        opRGB,
        LMS,
        HSV
        # PIGMENT EXCLUDED
        # MATRIX EXCLUDED
    ]
    """
    List of nodes representing types in the transformation graph
    """

    __default_transformations: dict[tuple[Type[ParentType], Type[ParentType]], Callable[[str], str]] = {
        # Format
        # Key: Tuple of origin space to destination space
        # Value: Transformation function that takes in an object and returns the generated onnx node's output name

        # intra color type transformations
        (XYZ, sRGB): color_transformation_generator(
            XYZ,
            sRGB,
            np.asarray([
                [3.2404542, -1.5371385, -0.4985314],
                [-0.9692660, 1.8760108, 0.0415560],
                [0.0556434, -0.2040259, 1.0572252]
            ]).transpose()
        ),
        (sRGB, XYZ): color_transformation_generator(
            sRGB,
            XYZ,
            np.asarray([
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041]
            ]).transpose()
        ),
        (opRGB, XYZ): color_transformation_generator(
            opRGB,
            XYZ,
            np.asarray([
                [0.57667, 0.29734, 0.02703],
                [0.18556, 0.62736, 0.07069],
                [0.18823, 0.07529, 0.99134]
            ])
        ),
        (XYZ, opRGB): color_transformation_generator(
            XYZ,
            opRGB,
            np.asarray([
                [2.04159, -0.96924, 0.01344],
                [-0.56501, 1.87597, -0.11836],
                [-0.34473, 0.04156, 1.01517]
            ])
        ),
        (LMS, XYZ): color_transformation_generator(
            LMS,
            XYZ,
            np.asarray([
                [1.94735469, 0.68990272, 0],
                [-1.41445123, 0.34832189, 0],
                [0.36476327, 0, 1.93485343]
            ])
        ),
        (XYZ, LMS): color_transformation_generator(
            XYZ,
            LMS,
            np.asarray([
                [0.210576, -0.417076, 0],
                [0.855098, 1.17726, 0],
                [-0.0396983, 0.0786283, 0.516835]
            ])
        ),
        (LightSpectrum, LMS): TransformationFunctions.light_spectrum_to_lms,

        # Chromaticity
        (XYZ, Chromaticity): TransformationFunctions.xyz_to_chromaticity,

        # Transformations to perceptual color spaces
        (XYZ, LAB): TransformationFunctions.xyz_to_lab,
        (LAB, XYZ): TransformationFunctions.lab_to_xyz,

        (sRGB, HSV): TransformationFunctions.srgb_to_hsv,
        (HSV, sRGB): TransformationFunctions.hsv_to_srgb

        # Transformations to matrix type needs to be handled separately
        # This is because we do not want to use multiple edges for transformation from/to matrices
        # Pigment is excluded, as the data it stores is not concrete
    }
    """
    Map of default transformations
    """

    @staticmethod
    def __create_populated_transformation_graph(
            nodes: list[Type[ParentType]],
            edges: dict[tuple[Type[ParentType], Type[ParentType]], Callable[[str], str]]) -> nx.DiGraph:
        """
        Populates the static transformation graph with the correct nodes
        """
        g = nx.DiGraph()

        # Populate nodes
        g.add_nodes_from(nodes)

        # Populate edges
        g.add_edges_from(
            [(origin_space, dest_space, {"func": trans_func}) for (origin_space, dest_space), trans_func in edges.items()]
        )

        return g

    __transformation_graph = __create_populated_transformation_graph(__node_list, __default_transformations)
    """
    Graph used to check validity of transformations
    """

    @classmethod
    def check_transformation_validity(cls, origin_class: Type[ParentType], destination_class: Type[ParentType]):
        # Special cases: Matrix origin/destination
        if origin_class == Matrix:
            return destination_class._matrixable
        elif destination_class == Matrix:
            return origin_class._matrixable

        return nx.has_path(cls.__transformation_graph, origin_class, destination_class)

    @classmethod
    def construct_transformation_onnx_nodes(cls, origin_object: ParentType, destination_class: Type[ParentType]) -> str:
        """
        Constructs onnx transformation nodes. Assumes that the transformation is valid.
        """
        # Special cases: Matrix origin/destination
        if origin_object.__class__ == Matrix or destination_class == Matrix:
            return origin_object._onnx_name

        # Find the shortest path
        shortest_path = nx.shortest_path(cls.__transformation_graph, origin_object.__class__, destination_class)

        # For each edge, apply the corresponding transformation function
        next_input = origin_object._onnx_name
        for edge_origin, edge_destination in zip(shortest_path, shortest_path[1:]):
            transformation_function = cls.__transformation_graph[edge_origin][edge_destination]["func"]
            next_input = transformation_function(next_input)

        return next_input
# endregion


# region Operation logic
class MetaOperationFunctions:
    """
    Operation implementation that require knowledge of types
    """

    @staticmethod
    def binary_color_matrix_multiplication(left_operand: ParentType, right_operand: ParentType):
        # Check ordering of types
        if isinstance(left_operand, ColorType):
            color_obj = left_operand
            matrix_obj = right_operand
        else:
            color_obj = right_operand
            matrix_obj = left_operand

        # Special case for gamma
        if color_obj.gamma is not None:
            # Gamma constant
            remove_gamma_constant = OnnxConstructor.create_constant(np.asarray([left_operand.gamma]))
            # Bit depth constant
            bit_depth_constant = OnnxConstructor.create_constant(np.asarray([2 ** left_operand.bit_depth - 1]))

            # Remove bit depth
            removed_bit_depth = OnnxConstructor.create_operation(
                op_type="Div",
                inputs=[color_obj._onnx_name, bit_depth_constant]
            )

            # Remove gamma
            removed_gamma = OnnxConstructor.create_operation(
                op_type="Pow",
                inputs=[removed_bit_depth, remove_gamma_constant]
            )

            # Multiply
            multiplied = OnnxConstructor.create_operation(
                op_type="Mul",
                inputs=[removed_gamma, matrix_obj._onnx_name]
            )

            # Apply gamma constant
            one = OnnxConstructor.create_constant(np.asarray([1]))
            apply_gamma_constant = OnnxConstructor.create_operation(
                op_type="Div",
                inputs=[one, remove_gamma_constant]
            )
            reapplied_gamma = OnnxConstructor.create_operation(
                op_type="Pow",
                inputs=[multiplied, apply_gamma_constant]
            )

            # Apply bit depth
            return OnnxConstructor.create_operation(
                op_type="Mul",
                inputs=[reapplied_gamma, bit_depth_constant]
            )

        # Normal case
        return OnnxConstructor.create_operation(
            op_type="Mul",
            inputs=[color_obj._onnx_name, matrix_obj._onnx_name]
        )


class OperationIdentifier:
    """
    Class that is used to match a list of operand orders to a concrete operation
    """
    def __init__(self, operand_type_list: list[Type[ParentType]], order_matters: bool):
        self.operand_type_list = operand_type_list
        self.order_matters = order_matters
        pass

    def match(self, operand_type_list: list[Type[ParentType]]):
        # If order matters, we need to check ordered list equality
        if self.order_matters:
            return self.operand_type_list == operand_type_list

        # If order does not matter, we need to check unordered equality
        matching_list = operand_type_list.copy()
        reference_list = self.operand_type_list.copy()
        for item in matching_list:
            if item not in reference_list:
                return False
            reference_list.remove(item)

        if len(reference_list) != 0:
            return False

        return True


class OperationManager:
    """
    Singleton class to check operation validity and to handle creation of corresponding onnx nodes
    """
    class SemanticOperationEnum(Enum):
        """
        Enum for semantic operation types (i.e. ones that are expressed in user's code)
        """
        ADD = "+",
        MUL = "*",
        DIV = "/",
        SUB = "-",
        MATMUL = "x",

    class ConcreteOperation:
        """
        Used to store metadata about concrete operations
        """
        def __init__(self,
                     shape_validation_function: Callable[[ParentType, ParentType], bool],
                     construction_function: Callable[[ParentType, ParentType], str],
                     output_type: Type[ParentType]):
            self.shape_validation_function = shape_validation_function
            self.construction_function = construction_function
            self.output_type = output_type

    __operation_map = {
        SemanticOperationEnum.MUL: {
            # Reflecting light off of a known reflectance spectrum
            OperationIdentifier([LightSpectrum, ReflectanceSpectrum], False): ConcreteOperation(
                shape_validation_function=ShapeValidationFunctions.binary_exact_match,
                construction_function=OperationFunctions.binary_elementwise_multiplication,
                output_type=LightSpectrum
            ),
            # Elementwise multiplication
            OperationIdentifier([Matrix, Matrix], False): ConcreteOperation(
                shape_validation_function=ShapeValidationFunctions.broadcastable_match,
                construction_function=OperationFunctions.binary_elementwise_multiplication,
                output_type=Matrix
            ),
            # Elementwise multiplication with scaling
            OperationIdentifier([ColorType, Matrix], False): ConcreteOperation(
                shape_validation_function=ShapeValidationFunctions.broadcastable_match,
                construction_function=MetaOperationFunctions.binary_color_matrix_multiplication,
                output_type=ColorType
            )
        },
        SemanticOperationEnum.ADD: {
            # Simple addition of colors
            OperationIdentifier([ColorType, ColorType], False): ConcreteOperation(
                shape_validation_function=ShapeValidationFunctions.binary_exact_match,
                construction_function=OperationFunctions.binary_color_addition,
                output_type=ColorType
            ),
            # Addition of light spectra
            OperationIdentifier([LightSpectrum, LightSpectrum], False): ConcreteOperation(
                shape_validation_function=ShapeValidationFunctions.binary_exact_match,
                construction_function=OperationFunctions.binary_elementwise_addition,
                output_type=ColorType
            ),
            # Matrix element-wise addition
            OperationIdentifier([Matrix, Matrix], False): ConcreteOperation(
                shape_validation_function=ShapeValidationFunctions.broadcastable_match,
                construction_function=OperationFunctions.binary_elementwise_addition,
                output_type=Matrix
            ),
            # Addition of LAB
            OperationIdentifier([LAB, LAB], False): ConcreteOperation(
                shape_validation_function=ShapeValidationFunctions.binary_exact_match,
                construction_function=OperationFunctions.binary_elementwise_addition,
                output_type=LAB
            ),
            # Addition of HSV
            OperationIdentifier([HSV, HSV], False): ConcreteOperation(
                shape_validation_function=ShapeValidationFunctions.binary_exact_match,
                construction_function=OperationFunctions.binary_elementwise_addition,
                output_type=HSV
            ),
        },
        SemanticOperationEnum.DIV: {
            # Matrix element-wise division
            OperationIdentifier([Matrix, Matrix], False): ConcreteOperation(
                shape_validation_function=ShapeValidationFunctions.broadcastable_match,
                construction_function=OperationFunctions.binary_elementwise_division,
                output_type=Matrix
            )
        },
        SemanticOperationEnum.SUB: {
            OperationIdentifier([Matrix, Matrix], False): ConcreteOperation(
                shape_validation_function=ShapeValidationFunctions.broadcastable_match,
                construction_function=OperationFunctions.binary_elementwise_subtraction,
                output_type=Matrix
            ),
            OperationIdentifier([LAB, LAB], False): ConcreteOperation(
                shape_validation_function=ShapeValidationFunctions.binary_exact_match,
                construction_function=OperationFunctions.binary_elementwise_subtraction,
                output_type=Matrix
            ),
        },
        SemanticOperationEnum.MATMUL: {
            OperationIdentifier([Matrix, Matrix], False): ConcreteOperation(
                shape_validation_function=ShapeValidationFunctions.matrix_multiplication_match,
                construction_function=OperationFunctions.binary_matrix_multiplication,
                output_type=Matrix
            ),
            OperationIdentifier([ColorType, Matrix], True): ConcreteOperation(
                shape_validation_function=ShapeValidationFunctions.matrix_multiplication_match,
                construction_function=OperationFunctions.binary_matrix_multiplication,
                output_type=ColorType
            )
        }

    }

    @classmethod
    def construct_operation(cls,
                            semantic_op: SemanticOperationEnum,
                            left_operand: ParentType,
                            right_operand: ParentType) -> ParentType:
        """
        Checks validity of operation. If valid, constructs operation nodes. Otherwise, throw error.
        """
        # If either operand is a color type, we use color type to search as opposed to their concrete type
        left_operand_class = left_operand.__class__ if not isinstance(left_operand, ColorType) else ColorType
        right_operand_class = right_operand.__class__ if not isinstance(right_operand, ColorType) else ColorType

        # Important edge case: if both operands are color types, their concrete types must match
        if left_operand_class == ColorType and right_operand_class == ColorType and left_operand.__class__ != right_operand.__class__:
            raise OperationNotFoundError(
                "The requested {} operation between {} and {} does not exist".format(
                    semantic_op.name,
                    left_operand.__class__,
                    right_operand.__class__
                )
            )

        # We must memorize the original color type
        color_output_type = None
        if isinstance(left_operand, ColorType):
            color_output_type = left_operand.__class__
        elif isinstance(right_operand, ColorType):
            color_output_type = right_operand.__class__

        # Extract concrete operation metadata by iterating through each semantic operation dictionary to check match
        concrete_operation: Optional[cls.ConcreteOperation] = None
        for identifier, operation in cls.__operation_map[semantic_op].items():
            if identifier.match([left_operand_class, right_operand_class]):
                concrete_operation = operation
                break

        if concrete_operation is None:
            raise OperationNotFoundError(
               "The requested {} operation between {} and {} does not exist".format(
                   semantic_op.name,
                   left_operand_class,
                   right_operand_class
               )
            )
        # Shape validation
        if not concrete_operation.shape_validation_function(left_operand, right_operand):
            raise OperandShapeMismatchError()

        # At this point, we know that the operation is valid
        # Perform operation construction here
        output_onnx_name = concrete_operation.construction_function(left_operand, right_operand)

        # Construct colorspace object
        output_type = concrete_operation.output_type
        # ColorType must be replaced with actual concrete type
        if output_type == ColorType:
            assert output_type is not None
            output_type = color_output_type

        onnx_shape = OnnxConstructor.get_shape(output_onnx_name)
        color_shape = output_type._onnx_shape_to_color_shape(onnx_shape)

        constructed_colorspace_object = output_type(
            onnx_name=output_onnx_name,
            onnx_shape=onnx_shape,
            color_shape=color_shape,
            operand_list=[left_operand, right_operand],
            color_operator=concrete_operation
        )

        return constructed_colorspace_object

# endregion

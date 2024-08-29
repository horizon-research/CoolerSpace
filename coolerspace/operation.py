from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import spaces

import numpy as np

from .construction import OnnxConstructor


class OperationFunctions:
    """
    Container class for operation construction functions.
    Operation construction functions take the onnx name of the operands as input, and provides the onnx name of the operation node as the output
    """
    @staticmethod
    def binary_elementwise_addition(left_operand: spaces.ParentType, right_operand: spaces.ParentType):
        output_string = OnnxConstructor.create_operation(
            "Add",
            [left_operand._onnx_name, right_operand._onnx_name]
        )
        return output_string

    @staticmethod
    def binary_elementwise_multiplication(left_operand: spaces.ParentType, right_operand: spaces.ParentType):
        output_string = OnnxConstructor.create_operation(
            "Mul",
            [left_operand._onnx_name, right_operand._onnx_name]
        )
        return output_string

    @staticmethod
    def binary_elementwise_division(left_operand: spaces.ParentType, right_operand: spaces.ParentType):
        return OnnxConstructor.create_operation(
            "Div",
            [left_operand._onnx_name, right_operand._onnx_name]
        )

    @staticmethod
    def binary_matrix_multiplication(left_operand: spaces.ParentType, right_operand: spaces.ParentType):
        output_string = OnnxConstructor.create_operation(
            "MatMul",
            [left_operand._onnx_name, right_operand._onnx_name]
        )
        return output_string

    @staticmethod
    def binary_elementwise_subtraction(left_operand: spaces.ParentType, right_operand: spaces.ParentType):
        output_string = OnnxConstructor.create_operation(
            op_type="Sub",
            inputs=[left_operand._onnx_name, right_operand._onnx_name]
        )
        return output_string

    @staticmethod
    def binary_color_addition(left_operand: spaces.ColorType, right_operand: spaces.ColorType):
        left_operand_name = left_operand._onnx_name
        right_operand_name = right_operand._onnx_name

        # The operands must be of the same type, and must be color types
        # Should never happen to a user
        assert left_operand.__class__ == right_operand.__class__

        # Remove gamma if applicable
        if left_operand.gamma is not None:
            # Gamma constant
            remove_gamma_constant = OnnxConstructor.create_constant(np.asarray([left_operand.gamma]))
            # Bit depth constant
            bit_depth_constant = OnnxConstructor.create_constant(np.asarray([2 ** left_operand.bit_depth - 1]))

            # Remove bit depth
            left_operand_name = OnnxConstructor.create_operation(
                op_type="Div",
                inputs=[left_operand_name, bit_depth_constant]
            )
            right_operand_name = OnnxConstructor.create_operation(
                op_type="Div",
                inputs=[right_operand_name, bit_depth_constant]
            )

            # Remove gamma
            left_operand_name = OnnxConstructor.create_operation(
                op_type="Pow",
                inputs=[left_operand_name, remove_gamma_constant]
            )
            right_operand_name = OnnxConstructor.create_operation(
                op_type="Pow",
                inputs=[right_operand_name, remove_gamma_constant]
            )

        # Elementwise addition
        final_sum = OnnxConstructor.create_operation(
            op_type="Add",
            inputs=[left_operand_name, right_operand_name]
        )

        # Reapply gamma if applicable
        if left_operand.gamma is not None:
            # Gamma application constant
            add_gamma_constant = OnnxConstructor.create_constant(np.asarray([1 / left_operand.gamma]))
            # Bit depth constant
            bit_depth_constant = OnnxConstructor.create_constant(np.asarray([2 ** left_operand.bit_depth - 1]))

            # Remove gamma
            final_sum = OnnxConstructor.create_operation(
                op_type="Pow",
                inputs=[final_sum, add_gamma_constant]
            )

            # Re-apply bit depth
            final_sum = OnnxConstructor.create_operation(
                op_type="Mul",
                inputs=[final_sum, bit_depth_constant]
            )

        return final_sum

    @staticmethod
    def calculate_reflectance_from_absorption_scattering(
            absorption_onnx_name: str,
            scattering_onnx_name: str):

        ar = OnnxConstructor.create_operation(
            op_type="Div",
            inputs=[absorption_onnx_name, scattering_onnx_name]
        )

        return OperationHelpers.calculate_reflectance_from_ar(ar)


class OperationHelpers:
    """
    Contains helper intermediary functions for operation construction methods.
    """

    @staticmethod
    def normalize_matrix_list(matrix_list: list[spaces.Matrix]):
        running_sum = matrix_list[0]._onnx_name

        for i in range(1, len(matrix_list)):
            running_sum = OnnxConstructor.create_operation(
                op_type="Add",
                inputs=[running_sum, matrix_list[i]._onnx_name]
            )

        return [
            OnnxConstructor.create_operation(
                op_type="Div",
                inputs=[matrix._onnx_name, running_sum]
            ) for matrix in matrix_list
        ]

    @staticmethod
    def ks_weighted(weights: list[spaces.Matrix], absorption_spectra: list[str], scattering_spectra: list[str]) -> tuple[str, str]:
        """
        Calculates weighted K/S given a list of weights, absorption spectra, and scattering spectra
        """
        # Normalize weights
        onnx_weights = OperationHelpers.normalize_matrix_list(weights)

        # Calculate weighted absorption and scattering spectra
        weighted_absorption = [
            OnnxConstructor.create_operation(
                op_type="Mul",
                inputs=[weight, absorption_spectrum]
            )
            for weight, absorption_spectrum in zip(onnx_weights, absorption_spectra)
        ]
        weighted_scattering = [
            OnnxConstructor.create_operation(
                op_type="Mul",
                inputs=[weight, scattering_spectrum]
            )
            for weight, scattering_spectrum in zip(onnx_weights, scattering_spectra)
        ]

        # Sum up totals
        total_weighted_absorption = OnnxConstructor.create_operation(
            op_type="Add",
            inputs=[weighted_absorption[0], weighted_absorption[1]]
        )
        for absorption in weighted_absorption[2:]:
            total_weighted_absorption = OnnxConstructor.create_operation(
                op_type="Add",
                inputs=[total_weighted_absorption, absorption]
            )

        total_weighted_scattering = OnnxConstructor.create_operation(
            op_type="Add",
            inputs=[weighted_scattering[0], weighted_scattering[1]]
        )
        for scattering in weighted_scattering[2:]:
            total_weighted_scattering = OnnxConstructor.create_operation(
                op_type="Add",
                inputs=[total_weighted_scattering, scattering]
            )

        return total_weighted_absorption, total_weighted_scattering

    @staticmethod
    def calculate_reflectance_from_ar(ar: str):
        ar_squared = OnnxConstructor.create_operation(
            op_type="Mul",
            inputs=[ar, ar]
        )

        ar_double = OnnxConstructor.create_operation(
            op_type="Add",
            inputs=[ar, ar]
        )

        ar_squared_plus_double = OnnxConstructor.create_operation(
            op_type="Add",
            inputs=[ar_squared, ar_double]
        )

        half = OnnxConstructor.create_constant(
            value=np.asarray([1 / 2])
        )

        ar_squared_plus_double_sqrt = OnnxConstructor.create_operation(
            op_type="Pow",
            inputs=[ar_squared_plus_double, half]
        )

        one = OnnxConstructor.create_constant(
            value=np.asarray([1])
        )

        ar_plus_one = OnnxConstructor.create_operation(
            op_type="Add",
            inputs=[ar, one]
        )

        final = OnnxConstructor.create_operation(
            op_type="Sub",
            inputs=[ar_plus_one, ar_squared_plus_double_sqrt]
        )

        return final



class ShapeValidationFunctions:
    """
    Container class for shape validation functions.
    Shape validation functions operate on shapes, and return boolean indicating validity of operation.
    """
    @staticmethod
    def binary_exact_match(left_operand: spaces.ParentType, right_operand: spaces.ParentType):
        return left_operand._color_shape == right_operand._color_shape

    @staticmethod
    def matrix_multiplication_match(left_operand: spaces.ParentType, right_operand: spaces.ParentType):
        return left_operand._onnx_shape[-1] == right_operand._onnx_shape[0]

    @staticmethod
    def broadcastable_match(left_operand: spaces.ParentType, right_operand: spaces.ParentType):
        left_operand_shape = left_operand._onnx_shape.copy()
        right_operand_shape = right_operand._onnx_shape.copy()

        left_operand_shape.reverse()
        right_operand_shape.reverse()

        for i in range(min(len(left_operand_shape), len(right_operand_shape))):
            if left_operand_shape[i] == 1 or right_operand_shape[i] == 1:
                # By broadcasting rules, this is sound
                pass
            elif left_operand_shape[i] != right_operand_shape[i]:
                return False

        return True
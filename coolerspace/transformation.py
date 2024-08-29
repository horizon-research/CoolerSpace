from __future__ import annotations
from typing import TYPE_CHECKING

from onnx.reference.op_run import OnnxType

if TYPE_CHECKING:
    import spaces

import numpy as np
import onnx
from typing import Callable, Type

from .construction import OnnxConstructor


# region Functions
class TransformationFunctions:
    """
    Container class for transformation functions.
    Transformation functions take the original colorspace object as input, and returns the onnx output name.
    """
    @staticmethod
    def xyz_to_chromaticity(xyz: str) -> str:
        """
        ONNX constructor for XYZ to Chromaticity
        """
        # Sum up XYZ total
        sum_matrix = OnnxConstructor.create_constant(
            value=np.asarray([
                [1],
                [1],
                [1]
            ])
        )
        xyz_total = OnnxConstructor.create_operation(
            op_type="MatMul",
            inputs=[xyz, sum_matrix]
        )

        # Casting XYZ to XY
        xyz_to_xy = OnnxConstructor.create_constant(
            value=np.asarray([
                [1, 0],
                [0, 1],
                [0, 0]
            ])
        )
        xy = OnnxConstructor.create_operation(
            op_type="MatMul",
            inputs=[xyz, xyz_to_xy]
        )

        # Get chromaticity
        return OnnxConstructor.create_operation(
            op_type="Div",
            inputs=[xy, xyz_total]
        )


    @staticmethod
    def nothing(name: str) -> str:
        """
        Does nothing
        """
        return name

    @staticmethod
    def xyz_to_lab(xyz: str) -> str:
        """
        ONNX constructor for XYZ to LAB
        """
        # D65 constant
        d65 = OnnxConstructor.create_constant(
            value=np.asarray([0.950489, 1, 1.088840])
        )
        # Calculate t
        t = OnnxConstructor.create_operation(
            op_type="Div",
            inputs=[xyz, d65]
        )
        # Delta constants
        delta = OnnxConstructor.create_constant(
            value=np.asarray([6/29])
        )
        # Left linear portion
        negative_one = OnnxConstructor.create_constant(
            value=np.asarray([-1])
        )
        m_constant = OnnxConstructor.create_constant(
            value=np.asarray([-(1 / (3 * ((6/29) ** 2)))])
        )
        k_constant = OnnxConstructor.create_constant(
            value=np.asarray([2 / 29])
        )
        negative_mt = OnnxConstructor.create_operation(
            op_type="Mul",
            inputs=[t, m_constant]
        )
        negative_mtk = OnnxConstructor.create_operation(
            op_type="Add",
            inputs=[negative_mt, k_constant]
        )
        negative_left_portion = OnnxConstructor.create_operation(
            op_type="Relu",
            inputs=[negative_mtk]
        )
        left_portion = OnnxConstructor.create_operation(
            op_type="Mul",
            inputs=[negative_left_portion, negative_one]
        )
        # Right nonlinear portion
        third = OnnxConstructor.create_constant(
            value=np.asarray([1/3])
        )
        cube_root_t = OnnxConstructor.create_operation(
            op_type="Pow",
            inputs=[t, third]
        )
        cube_root_minus_d = OnnxConstructor.create_operation(
            op_type="Sub",
            inputs=[cube_root_t, delta]
        )
        right_portion = OnnxConstructor.create_operation(
            op_type="Relu",
            inputs=[cube_root_minus_d]
        )
        # Combine
        total = OnnxConstructor.create_operation(
            op_type="Add",
            inputs=[left_portion, right_portion]
        )
        total_plus_d = OnnxConstructor.create_operation(
            op_type="Add",
            inputs=[total, delta]
        )
        # Calculating lab
        matrix = OnnxConstructor.create_constant(
            value=np.asarray([
                [0, 500, 0],
                [116, -500, 200],
                [0, 0, -200]
            ])
        )
        final_matrix_constant = OnnxConstructor.create_constant(
            value=np.asarray([
                [-16, 0, 0]
            ])
        )
        lab_before_l_constant_removal = OnnxConstructor.create_operation(
            op_type="MatMul",
            inputs=[total_plus_d, matrix]
        )
        lab = OnnxConstructor.create_operation(
            op_type="Add",
            inputs=[lab_before_l_constant_removal, final_matrix_constant]
        )
        return lab

    @staticmethod
    def lab_to_xyz(lab: str) -> str:
        """
        ONNX constructor for lab to xyz
        """
        # Delta
        negative_delta_cubed = OnnxConstructor.create_constant(
            value=np.asarray([- (6/29) ** 3])
        )
        delta_cubed = OnnxConstructor.create_constant(
            value=np.asarray([(6/29) ** 3])
        )
        # Calculating t
        l_offset = OnnxConstructor.create_constant(
            value=np.asarray([16, 0, 0])
        )
        adjusted_lab = OnnxConstructor.create_operation(
            op_type="Add",
            inputs=[lab, l_offset]
        )
        lab_to_t_matrix = OnnxConstructor.create_constant(
            value=np.asarray([
                [1/116, 1/116, 1/116],
                [1/500, 0, 0],
                [0, 0, -1/200]
            ])
        )
        t = OnnxConstructor.create_operation(
            op_type="MatMul",
            inputs=[adjusted_lab, lab_to_t_matrix]
        )
        # Calculate f-1(t)
        # Right hand portion
        three = OnnxConstructor.create_constant(
            value=np.asarray([3])
        )
        t_cubed = OnnxConstructor.create_operation(
            op_type="Pow",
            inputs=[t, three]
        )
        t_cubed_minus_d = OnnxConstructor.create_operation(
            op_type="Add",
            inputs=[t_cubed, negative_delta_cubed]
        )
        right_portion = OnnxConstructor.create_operation(
            op_type="Relu",
            inputs=[t_cubed_minus_d]
        )
        # Left hand portion
        m_constant = OnnxConstructor.create_constant(
            value=np.asarray([
                -3 * ((6 / 29) ** 2)
            ])
        )
        k_constant = OnnxConstructor.create_constant(
            value=np.asarray([
                -((3 * ((6 / 29) ** 2)) * (-4/29) - ((6/29) ** 3))
            ])
        )
        mt = OnnxConstructor.create_operation(
            op_type="Mul",
            inputs=[t, m_constant]
        )
        negative_mtk = OnnxConstructor.create_operation(
            op_type="Add",
            inputs=[mt, k_constant]
        )
        negative_one = OnnxConstructor.create_constant(
            value=np.asarray([-1])
        )
        negative_left_portion = OnnxConstructor.create_operation(
            op_type="Relu",
            inputs=[negative_mtk]
        )
        left_portion = OnnxConstructor.create_operation(
            op_type="Mul",
            inputs=[negative_left_portion, negative_one]
        )
        # Combine
        total = OnnxConstructor.create_operation(
            op_type="Add",
            inputs=[left_portion, right_portion]
        )
        total_plus_constant = OnnxConstructor.create_operation(
            op_type="Add",
            inputs=[total, delta_cubed]
        )
        # Scaling by D65 values
        d65 = OnnxConstructor.create_constant(
            value=np.asarray([0.950489, 1, 1.088840])
        )
        xyz = OnnxConstructor.create_operation(
            op_type="Mul",
            inputs=[total_plus_constant, d65]
        )
        return xyz

    @staticmethod
    def light_spectrum_to_lms(light_spectrum: str) -> str:
        """
        Onnx constructor for Light Spectrum to LMS
        """
        spectrum_to_lms = OnnxConstructor.create_constant(
            value=np.asarray(
                [
                    [4.15003E-04, 3.68349E-04, 9.54729E-03],
                    [1.05192E-03, 9.58658E-04, 2.38250E-02],
                    [2.40836E-03, 2.26991E-03, 5.66498E-02],
                    [4.83339E-03, 4.70010E-03, 1.22451E-01],
                    [8.72127E-03, 8.79369E-03, 2.33008E-01],
                    [1.33837E-02, 1.45277E-02, 3.81363E-01],
                    [1.84480E-02, 2.16649E-02, 5.43618E-01],
                    [2.29317E-02, 2.95714E-02, 6.74474E-01],
                    [2.81877E-02, 3.94566E-02, 8.02555E-01],
                    [3.41054E-02, 5.18199E-02, 9.03573E-01],
                    [4.02563E-02, 6.47782E-02, 9.91020E-01],
                    [4.49380E-02, 7.58812E-02, 9.91515E-01],
                    [4.98639E-02, 8.70524E-02, 9.55393E-01],
                    [5.53418E-02, 9.81934E-02, 8.60240E-01],
                    [6.47164E-02, 1.16272E-01, 7.86704E-01],
                    [8.06894E-02, 1.44541E-01, 7.38268E-01],
                    [9.94755E-02, 1.75893E-01, 6.46359E-01],
                    [1.18802E-01, 2.05398E-01, 5.16411E-01],
                    [1.40145E-01, 2.35754E-01, 3.90333E-01],
                    [1.63952E-01, 2.68063E-01, 2.90322E-01],
                    [1.91556E-01, 3.03630E-01, 2.11867E-01],
                    [2.32926E-01, 3.57061E-01, 1.60526E-01],
                    [2.88959E-01, 4.27764E-01, 1.22839E-01],
                    [3.59716E-01, 5.15587E-01, 8.88965E-02],
                    [4.43683E-01, 6.15520E-01, 6.08210E-02],
                    [5.36494E-01, 7.19154E-01, 4.28123E-02],
                    [6.28561E-01, 8.16610E-01, 2.92033E-02],
                    [7.04720E-01, 8.85550E-01, 1.93912E-02],
                    [7.70630E-01, 9.35687E-01, 1.26013E-02],
                    [8.25711E-01, 9.68858E-01, 8.09453E-03],
                    [8.81011E-01, 9.95217E-01, 5.08900E-03],
                    [9.19067E-01, 9.97193E-01, 3.16893E-03],
                    [9.40198E-01, 9.77193E-01, 1.95896E-03],
                    [9.65733E-01, 9.56583E-01, 1.20277E-03],
                    [9.81445E-01, 9.17750E-01, 7.40174E-04],
                    [9.94486E-01, 8.73205E-01, 4.55979E-04],
                    [9.99993E-01, 8.13509E-01, 2.81800E-04],
                    [9.92310E-01, 7.40291E-01, 1.75039E-04],
                    [9.69429E-01, 6.53274E-01, 1.09454E-04],
                    [9.55602E-01, 5.72597E-01, 6.89991E-05],
                    [9.27673E-01, 4.92599E-01, 4.39024E-05],
                    [8.85969E-01, 4.11246E-01, 2.82228E-05],
                    [8.33982E-01, 3.34429E-01, 1.83459E-05],
                    [7.75103E-01, 2.64872E-01, 1.20667E-05],
                    [7.05713E-01, 2.05273E-01, 8.03488E-06],
                    [6.30773E-01, 1.56243E-01, 5.41843E-06],
                    [5.54224E-01, 1.16641E-01, 0],
                    [4.79941E-01, 8.55872E-02, 0],
                    [4.00711E-01, 6.21120E-02, 0],
                    [3.27864E-01, 4.44879E-02, 0],
                    [2.65784E-01, 3.14282E-02, 0],
                    [2.13284E-01, 2.18037E-02, 0],
                    [1.65141E-01, 1.54480E-02, 0],
                    [1.24749E-01, 1.07120E-02, 0],
                    [9.30085E-02, 7.30255E-03, 0],
                    [6.85100E-02, 4.97179E-03, 0],
                    [4.98661E-02, 3.43667E-03, 0],
                    [3.58233E-02, 2.37617E-03, 0],
                    [2.53790E-02, 1.63734E-03, 0],
                    [1.77201E-02, 1.12128E-03, 0],
                    [1.21701E-02, 7.61051E-04, 0],
                    [8.47170E-03, 5.25457E-04, 0],
                    [5.89749E-03, 3.65317E-04, 0],
                    [4.09129E-03, 2.53417E-04, 0],
                    [2.80447E-03, 1.74402E-04, 0],
                    [1.92058E-03, 1.20608E-04, 0],
                    [1.32687E-03, 8.41716E-05, 0],
                    [9.17777E-04, 5.89349E-05, 0],
                    [6.39373E-04, 4.16049E-05, 0],
                    [4.46035E-04, 2.94354E-05, 0],
                    [3.10869E-04, 2.08860E-05, 0],
                    [2.19329E-04, 1.50458E-05, 0],
                    [1.54549E-04, 1.08200E-05, 0],
                    [1.09508E-04, 7.82271E-06, 0],
                    [7.79912E-05, 5.69093E-06, 0],
                    [5.56264E-05, 4.13998E-06, 0],
                    [3.99295E-05, 3.02683E-06, 0],
                    [2.86163E-05, 2.21100E-06, 0],
                    [2.07321E-05, 1.63433E-06, 0],
                    [1.50432E-05, 1.21054E-06, 0],
                    [1.09446E-05, 8.99170E-07, 0],
                    [7.97750E-06, 6.69594E-07, 0],
                    [5.85057E-06, 5.03187E-07, 0],
                    [4.31102E-06, 3.80046E-07, 0],
                    [3.17009E-06, 2.86329E-07, 0],
                    [2.34468E-06, 2.16878E-07, 0],
                    [1.74666E-06, 1.65158E-07, 0],
                    [1.30241E-06, 1.25508E-07, 0],
                    [9.74306E-07, 9.53411E-08, 0]
                ]
            )
        )
        lms = OnnxConstructor.create_operation(
            op_type="MatMul",
            inputs=[light_spectrum, spectrum_to_lms]
        )
        return lms

    @staticmethod
    def srgb_to_hsv(srgb: str) -> str:
        # Unapply bit depth
        srgb_max = OnnxConstructor.create_constant(
            value=np.asarray([255])
        )
        srgb = OnnxConstructor.create_operation(
            op_type="Div",
            inputs=[srgb, srgb_max]
        )

        # Extract maximum and minimum
        first_of_three = OnnxConstructor.create_constant(
            value=np.asarray([
                [1],
                [0],
                [0]
            ])
        )
        second_of_three = OnnxConstructor.create_constant(
            value=np.asarray([
                [0],
                [1],
                [0]
            ])
        )
        third_of_three = OnnxConstructor.create_constant(
            value=np.asarray([
                [0],
                [0],
                [1]
            ])
        )

        red = OnnxConstructor.create_operation(
            op_type="MatMul",
            inputs=[srgb, first_of_three]
        )
        green = OnnxConstructor.create_operation(
            op_type="MatMul",
            inputs=[srgb, second_of_three]
        )
        blue = OnnxConstructor.create_operation(
            op_type="MatMul",
            inputs=[srgb, third_of_three]
        )

        max = OnnxConstructor.create_operation(
            op_type="Max",
            inputs=[red, green, blue]
        )
        min = OnnxConstructor.create_operation(
            op_type="Min",
            inputs=[red, green, blue]
        )
        range = OnnxConstructor.create_operation(
            op_type="Sub",
            inputs=[max, min]
        )

        # Create boolean indicating if range value is zero
        negative_one = OnnxConstructor.create_constant(
            value=np.asarray([-1])
        )
        inverted_range = OnnxConstructor.create_operation(
            op_type="Mul",
            inputs=[range, negative_one]
        )
        inverted_range_floored = OnnxConstructor.create_operation(
            op_type="Floor",
            inputs=[inverted_range]
        )
        one = OnnxConstructor.create_constant(
            value=np.asarray([1])
        )
        inverted_range_floored_plus_one = OnnxConstructor.create_operation(
            op_type="Add",
            inputs=[inverted_range_floored, one]
        )
        is_range_zero = OnnxConstructor.create_operation(
            op_type="Relu",
            inputs=[inverted_range_floored_plus_one]
        )
        one_to_first_of_four = OnnxConstructor.create_constant(
            value=np.asarray([
                [1, 0, 0, 0]
            ])
        )

        is_range_zero_projected = OnnxConstructor.create_operation(
            op_type="MatMul",
            inputs=[is_range_zero, one_to_first_of_four]
        )

        # Create boolean indicating if max value is zero
        inverted_max = OnnxConstructor.create_operation(
            op_type="Mul",
            inputs=[max, negative_one]
        )
        inverted_max_floored = OnnxConstructor.create_operation(
            op_type="Floor",
            inputs=[inverted_max]
        )
        inverted_range_floored_plus_one = OnnxConstructor.create_operation(
            op_type="Add",
            inputs=[inverted_max_floored, one]
        )
        is_max_zero = OnnxConstructor.create_operation(
            op_type="Relu",
            inputs=[inverted_range_floored_plus_one]
        )
        is_max_zero_inverted = OnnxConstructor.create_operation(
            op_type="Mul",
            inputs=[is_max_zero, negative_one]
        )
        is_max_nonzero = OnnxConstructor.create_operation(
            op_type="Add",
            inputs=[is_max_zero_inverted, one]
        )

        # Range or nonzero allows us to divide by range or some non-zero value to avoid NaN
        nonzero_range_or_one = OnnxConstructor.create_operation(
            op_type="Add",
            inputs=[range, is_range_zero]
        )

        # Range or nonzero allows us to divide by max or some non-zero value to avoid NaN
        nonzero_max_or_one = OnnxConstructor.create_operation(
            op_type="Add",
            inputs=[max, is_max_zero]
        )

        # Extract max index
        max_plus_one = OnnxConstructor.create_operation(
            op_type="Add",
            inputs=[max, one]
        )
        srgb_plus_one = OnnxConstructor.create_operation(
            op_type="Add",
            inputs=[srgb, one]
        )
        srgb_normalized_by_max = OnnxConstructor.create_operation(
            op_type="Div",
            inputs=[srgb_plus_one, max_plus_one]
        )
        srgb_normalized_floored = OnnxConstructor.create_operation(
            op_type="Floor",
            inputs=[srgb_normalized_by_max]
        )
        three_to_last_three_of_four = OnnxConstructor.create_constant(
            value=np.asarray([
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        )
        srgb_normalized_projected = OnnxConstructor.create_operation(
            op_type="MatMul",
            inputs=[srgb_normalized_floored, three_to_last_three_of_four]
        )

        max_index_lookup_unprocessed = OnnxConstructor.create_operation(
            op_type="Add",
            inputs=[srgb_normalized_projected, is_range_zero_projected]
        )

        magic_matrix = OnnxConstructor.create_constant(
            value=np.asarray([
                [1, -1, -1, -1],
                [0, 1, -1, -1],
                [0, 0, 1, -1],
                [0, 0, 0, 1]
            ])
        )
        srgb_magic = OnnxConstructor.create_operation(
            op_type="MatMul",
            inputs=[max_index_lookup_unprocessed, magic_matrix]
        )
        max_index = OnnxConstructor.create_operation(
            op_type="Relu",
            inputs=[srgb_magic]
        )

        # Calculating piecewise hue
        green_minus_blue = OnnxConstructor.create_operation(
            op_type="Sub",
            inputs=[green, blue]
        )
        blue_minus_red = OnnxConstructor.create_operation(
            op_type="Sub",
            inputs=[blue, red]
        )
        red_minus_green = OnnxConstructor.create_operation(
            op_type="Sub",
            inputs=[red, green]
        )

        green_minus_blue_range = OnnxConstructor.create_operation(
            op_type="Div",
            inputs=[green_minus_blue, nonzero_range_or_one]
        )
        blue_minus_red_range = OnnxConstructor.create_operation(
            op_type="Div",
            inputs=[blue_minus_red, nonzero_range_or_one]
        )
        red_minus_green_range = OnnxConstructor.create_operation(
            op_type="Div",
            inputs=[red_minus_green, nonzero_range_or_one]
        )

        six = OnnxConstructor.create_constant(
            value=np.asarray([6])
        )
        hue_red_before_angle = OnnxConstructor.create_operation(
            op_type="Mod",
            inputs=[green_minus_blue_range, six],
            fmod=1
        )

        two = OnnxConstructor.create_constant(
            value=np.asarray([2])
        )
        hue_green_before_angle = OnnxConstructor.create_operation(
            op_type="Add",
            inputs=[blue_minus_red_range, two]
        )

        four = OnnxConstructor.create_constant(
            value=np.asarray([4])
        )
        hue_blue_before_angle = OnnxConstructor.create_operation(
            op_type="Add",
            inputs=[red_minus_green_range, four]
        )

        # Create hue lookup table
        second = OnnxConstructor.create_constant(
            value=np.asarray([
                [0, 1, 0, 0]
            ])
        )
        third = OnnxConstructor.create_constant(
            value=np.asarray([
                [0, 0, 1, 0]
            ])
        )
        fourth = OnnxConstructor.create_constant(
            value=np.asarray([
                [0, 0, 0, 1]
            ])
        )
        red_second = OnnxConstructor.create_operation(
            op_type="MatMul",
            inputs=[hue_red_before_angle, second]
        )
        green_third = OnnxConstructor.create_operation(
            op_type="MatMul",
            inputs=[hue_green_before_angle, third]
        )
        blue_fourth = OnnxConstructor.create_operation(
            op_type="MatMul",
            inputs=[hue_blue_before_angle, fourth]
        )

        red_green_second_third = OnnxConstructor.create_operation(
            op_type="Add",
            inputs=[red_second, green_third]
        )
        hue_lookup_table = OnnxConstructor.create_operation(
            op_type="Add",
            inputs=[red_green_second_third, blue_fourth]
        )

        # Lookup piecewise hue function based on index
        filtered_hues = OnnxConstructor.create_operation(
            op_type="Mul",
            inputs=[hue_lookup_table, max_index]
        )
        four_by_one_of_ones = OnnxConstructor.create_constant(
            value=np.asarray([
                [1],
                [1],
                [1],
                [1]
            ])
        )
        hue_before_degree = OnnxConstructor.create_operation(
            op_type="MatMul",
            inputs=[filtered_hues, four_by_one_of_ones]
        )
        sixty = OnnxConstructor.create_constant(
            value=np.asarray([60])
        )
        hue = OnnxConstructor.create_operation(
            op_type="Mul",
            inputs=[hue_before_degree, sixty]
        )

        # Saturation calculation
        saturation_before_filter = OnnxConstructor.create_operation(
            op_type="Div",
            inputs=[range, nonzero_max_or_one]
        )
        saturation = OnnxConstructor.create_operation(
            op_type="Mul",
            inputs=[saturation_before_filter, is_max_nonzero]
        )

        # Value is just maximum
        value = max

        # Combine all three
        project_one_to_three = OnnxConstructor.create_constant(
            value=np.asarray([[1, 0, 0]])
        )
        project_two_to_three = OnnxConstructor.create_constant(
            value=np.asarray([[0, 1, 0]])
        )
        project_three_to_three = OnnxConstructor.create_constant(
            value=np.asarray([[0, 0, 1]])
        )
        hue_adjusted = OnnxConstructor.create_operation(
            op_type="MatMul",
            inputs=[hue, project_one_to_three]
        )
        saturation_adjusted = OnnxConstructor.create_operation(
            op_type="MatMul",
            inputs=[saturation, project_two_to_three]
        )
        value_adjusted = OnnxConstructor.create_operation(
            op_type="MatMul",
            inputs=[value, project_three_to_three]
        )

        hue_saturation = OnnxConstructor.create_operation(
            op_type="Add",
            inputs=[hue_adjusted, saturation_adjusted]
        )
        hsv = OnnxConstructor.create_operation(
            op_type="Add",
            inputs=[hue_saturation, value_adjusted]
        )

        return hsv


    @staticmethod
    def hsv_to_srgb(hsv: str) -> str:
        # Extract h, s, v
        extract_h = OnnxConstructor.create_constant(
            value=np.asarray([
                [1],
                [0],
                [0]
            ])
        )
        extract_s = OnnxConstructor.create_constant(
            value=np.asarray([
                [0],
                [1],
                [0]
            ])
        )
        extract_v = OnnxConstructor.create_constant(
            value=np.asarray([
                [0],
                [0],
                [1]
            ])
        )
        hue = OnnxConstructor.create_operation(
            op_type="MatMul",
            inputs=[hsv, extract_h]
        )
        saturation = OnnxConstructor.create_operation(
            op_type="MatMul",
            inputs=[hsv, extract_s]
        )
        value = OnnxConstructor.create_operation(
            op_type="MatMul",
            inputs=[hsv, extract_v]
        )

        # Calculate Chroma
        two = OnnxConstructor.create_constant(
            value=np.asarray([2])
        )
        twice_val = OnnxConstructor.create_operation(
            op_type="Mul",
            inputs=[value, two]
        )

        one = OnnxConstructor.create_constant(
            value=np.asarray([1])
        )
        twice_sat_min_one = OnnxConstructor.create_operation(
            op_type="Sub",
            inputs=[twice_val, one]
        )
        twice_sat_min_one_abs = OnnxConstructor.create_operation(
            op_type="Abs",
            inputs=[twice_sat_min_one]
        )
        one_min_twice_sat_min_one_abs = OnnxConstructor.create_operation(
            op_type="Sub",
            inputs=[one, twice_sat_min_one_abs]
        )
        chroma = OnnxConstructor.create_operation(
            op_type="Mul",
            inputs=[one_min_twice_sat_min_one_abs, saturation]
        )

        # OnnxConstructor.create_output(chroma, OnnxConstructor.get_shape(chroma))

        # Intermediate value calculation
        num_60 = OnnxConstructor.create_constant(
            value=np.asarray([60])
        )
        hue_prime = OnnxConstructor.create_operation(
            op_type="Div",
            inputs=[hue, num_60]
        )
        hue_prime_mod_2 = OnnxConstructor.create_operation(
            op_type="Mod",
            inputs=[hue_prime, two],
            fmod=1
        )
        hue_prime_mod_2_min_1 = OnnxConstructor.create_operation(
            op_type="Sub",
            inputs=[hue_prime_mod_2, one]
        )
        hue_prime_mod_2_min_1_abs = OnnxConstructor.create_operation(
            op_type="Abs",
            inputs=[hue_prime_mod_2_min_1],
        )
        one_min_hue_prime_mode_2_min_1_abs = OnnxConstructor.create_operation(
            op_type="Sub",
            inputs=[one, hue_prime_mod_2_min_1_abs]
        )
        intermediate = OnnxConstructor.create_operation(
            op_type="Mul",
            inputs=[chroma, one_min_hue_prime_mode_2_min_1_abs]
        )

        # OnnxConstructor.create_output(intermediate, OnnxConstructor.get_shape(intermediate))

        # Calculate R', G', B' lookup table
        # rgb_prime_chroma_portion_broadcast = OnnxConstructor.create_constant(
        #     value=np.asarray([
        #         [1, 0, 0],
        #         [0, 1, 0],
        #         [0, 1, 0],
        #         [0, 0, 1],
        #         [0, 0, 1],
        #         [1, 0, 0]
        #     ])
        # )
        # print(OnnxConstructor.get_shape(rgb_prime_chroma_portion_broadcast))
        # rgb_prime_chroma_portion = OnnxConstructor.create_operation(
        #     op_type="Mul",
        #     inputs=[rgb_prime_chroma_portion_broadcast, chroma]
        # )
        # rgb_prime_intermediate_portion_broadcast = OnnxConstructor.create_constant(
        #     value=np.asarray([
        #         [0, 1, 0],
        #         [1, 0, 0],
        #         [0, 0, 1],
        #         [0, 1, 0],
        #         [1, 0, 0],
        #         [0, 0, 1]
        #     ])
        # )
        # rgb_prime_intermediate_portion = OnnxConstructor.create_operation(
        #     op_type="Mul",
        #     inputs=[rgb_prime_intermediate_portion_broadcast, intermediate]
        # )
        # rgb_prime_lookup = OnnxConstructor.create_operation(
        #     op_type="Add",
        #     inputs=[rgb_prime_chroma_portion, rgb_prime_intermediate_portion]
        # )

        # Calculate index onehot
        index_m1 = OnnxConstructor.create_operation(
            op_type="Floor",
            inputs=[hue_prime]
        )
        index = OnnxConstructor.create_operation(
            op_type="Add",
            inputs=[index_m1, one]
        )
        index_m2 = OnnxConstructor.create_operation(
            op_type="Sub",
            inputs=[index_m1, one]
        )
        index_m3 = OnnxConstructor.create_operation(
            op_type="Sub",
            inputs=[index_m2, one]
        )
        index_m4 = OnnxConstructor.create_operation(
            op_type="Sub",
            inputs=[index_m3, one]
        )
        index_m5 = OnnxConstructor.create_operation(
            op_type="Sub",
            inputs=[index_m4, one]
        )

        greater_0 = OnnxConstructor.create_operation(
            op_type="Relu",
            inputs=[index]
        )
        greater_1 = OnnxConstructor.create_operation(
            op_type="Relu",
            inputs=[index_m1]
        )
        greater_2 = OnnxConstructor.create_operation(
            op_type="Relu",
            inputs=[index_m2]
        )
        greater_3 = OnnxConstructor.create_operation(
            op_type="Relu",
            inputs=[index_m3]
        )
        greater_4 = OnnxConstructor.create_operation(
            op_type="Relu",
            inputs=[index_m4]
        )
        greater_5 = OnnxConstructor.create_operation(
            op_type="Relu",
            inputs=[index_m5]
        )

        greater_project_0 = OnnxConstructor.create_constant(
            value=np.asarray([[1, 0, 0, 0, 0, 0]])
        )
        greater_project_1 = OnnxConstructor.create_constant(
            value=np.asarray([[0, 1, 0, 0, 0, 0]])
        )
        greater_project_2 = OnnxConstructor.create_constant(
            value=np.asarray([[0, 0, 1, 0, 0, 0]])
        )
        greater_project_3 = OnnxConstructor.create_constant(
            value=np.asarray([[0, 0, 0, 1, 0, 0]])
        )
        greater_project_4 = OnnxConstructor.create_constant(
            value=np.asarray([[0, 0, 0, 0, 1, 0]])
        )
        greater_project_5 = OnnxConstructor.create_constant(
            value=np.asarray([[0, 0, 0, 0, 0, 1]])
        )

        greater_projected_0 = OnnxConstructor.create_operation(
            op_type="MatMul",
            inputs=[greater_0, greater_project_0]
        )
        greater_projected_1 = OnnxConstructor.create_operation(
            op_type="MatMul",
            inputs=[greater_1, greater_project_1]
        )
        greater_projected_2 = OnnxConstructor.create_operation(
            op_type="MatMul",
            inputs=[greater_2, greater_project_2]
        )
        greater_projected_3 = OnnxConstructor.create_operation(
            op_type="MatMul",
            inputs=[greater_3, greater_project_3]
        )
        greater_projected_4 = OnnxConstructor.create_operation(
            op_type="MatMul",
            inputs=[greater_4, greater_project_4]
        )
        greater_projected_5 = OnnxConstructor.create_operation(
            op_type="MatMul",
            inputs=[greater_5, greater_project_5]
        )

        onehot_01 = OnnxConstructor.create_operation(
            op_type="Add",
            inputs=[greater_projected_0, greater_projected_1]
        )
        onehot_012 = OnnxConstructor.create_operation(
            op_type="Add",
            inputs=[onehot_01, greater_projected_2]
        )
        onehot_0123 = OnnxConstructor.create_operation(
            op_type="Add",
            inputs=[onehot_012, greater_projected_3]
        )
        onehot_01234 = OnnxConstructor.create_operation(
            op_type="Add",
            inputs=[onehot_0123, greater_projected_4]
        )
        onehot_before_norm = OnnxConstructor.create_operation(
            op_type="Add",
            inputs=[onehot_01234, greater_projected_5]
        )

        num_6 = OnnxConstructor.create_constant(
            value=np.asarray([6])
        )
        onehot_norm = OnnxConstructor.create_operation(
            op_type="Div",
            inputs=[onehot_before_norm, num_6]
        )
        onehot = OnnxConstructor.create_operation(
            op_type="Ceil",
            inputs=[onehot_norm]
        )


        magic_matrix = OnnxConstructor.create_constant(
            value=np.asarray([
                [1, 0, 0, 0, 0, 0],
                [-1, 1, 0, 0, 0, 0],
                [-1, -1, 1, 0, 0, 0],
                [-1, -1, -1, 1, 0, 0],
                [-1, -1, -1, -1, 1, 0],
                [-1, -1, -1, -1, -1, 1]
            ])
        )

        onehot_magic = OnnxConstructor.create_operation(
            op_type="MatMul",
            inputs=[onehot, magic_matrix]
        )
        onehot = OnnxConstructor.create_operation(
            op_type="Relu",
            inputs=[onehot_magic]
        )

        # OnnxConstructor.create_output(onehot, OnnxConstructor.get_shape(onehot))

        # Get R', G', B'
        r_c_lookup = OnnxConstructor.create_constant(
            value=np.asarray([
                [1],
                [0],
                [0],
                [0],
                [0],
                [1]
            ])
        )
        r_int_lookup = OnnxConstructor.create_constant(
            value=np.asarray([
                [0],
                [1],
                [0],
                [0],
                [1],
                [0]
            ])
        )
        g_c_lookup = OnnxConstructor.create_constant(
            value=np.asarray([
                [0],
                [1],
                [1],
                [0],
                [0],
                [0]
            ])
        )
        g_int_lookup = OnnxConstructor.create_constant(
            value=np.asarray([
                [1],
                [0],
                [0],
                [1],
                [0],
                [0]
            ])
        )
        b_c_lookup = OnnxConstructor.create_constant(
            value=np.asarray([
                [0],
                [0],
                [0],
                [1],
                [1],
                [0]
            ])
        )
        b_int_lookup = OnnxConstructor.create_constant(
            value=np.asarray([
                [0],
                [0],
                [1],
                [0],
                [0],
                [1]
            ])
        )

        r_c_status = OnnxConstructor.create_operation(
            op_type="MatMul",
            inputs=[onehot, r_c_lookup]
        )
        g_c_status = OnnxConstructor.create_operation(
            op_type="MatMul",
            inputs=[onehot, g_c_lookup]
        )
        b_c_status = OnnxConstructor.create_operation(
            op_type="MatMul",
            inputs=[onehot, b_c_lookup]
        )
        r_int_status = OnnxConstructor.create_operation(
            op_type="MatMul",
            inputs=[onehot, r_int_lookup]
        )
        g_int_status = OnnxConstructor.create_operation(
            op_type="MatMul",
            inputs=[onehot, g_int_lookup]
        )
        b_int_status = OnnxConstructor.create_operation(
            op_type="MatMul",
            inputs=[onehot, b_int_lookup]
        )


        r_c_final = OnnxConstructor.create_operation(
            op_type="Mul",
            inputs=[r_c_status, chroma]
        )
        g_c_final = OnnxConstructor.create_operation(
            op_type="Mul",
            inputs=[g_c_status, chroma]
        )
        b_c_final = OnnxConstructor.create_operation(
            op_type="Mul",
            inputs=[b_c_status, chroma]
        )

        r_int_final = OnnxConstructor.create_operation(
            op_type="Mul",
            inputs=[r_int_status, intermediate]
        )
        g_int_final = OnnxConstructor.create_operation(
            op_type="Mul",
            inputs=[g_int_status, intermediate]
        )
        b_int_final = OnnxConstructor.create_operation(
            op_type="Mul",
            inputs=[b_int_status, intermediate]
        )

        r_final = OnnxConstructor.create_operation(
            op_type="Add",
            inputs=[r_c_final, r_int_final]
        )
        g_final = OnnxConstructor.create_operation(
            op_type="Add",
            inputs=[g_c_final, g_int_final]
        )
        b_final = OnnxConstructor.create_operation(
            op_type="Add",
            inputs=[b_c_final, b_int_final]
        )

        r_project = OnnxConstructor.create_constant(
            value=np.asarray([[1, 0, 0]])
        )
        g_project = OnnxConstructor.create_constant(
            value=np.asarray([[0, 1, 0]])
        )
        b_project = OnnxConstructor.create_constant(
            value=np.asarray([[0, 0, 1]])
        )

        r_comp = OnnxConstructor.create_operation(
            op_type="MatMul",
            inputs=[r_final, r_project]
        )
        g_comp = OnnxConstructor.create_operation(
            op_type="MatMul",
            inputs=[g_final, g_project]
        )
        b_comp = OnnxConstructor.create_operation(
            op_type="MatMul",
            inputs=[b_final, b_project]
        )


        tot_rg = OnnxConstructor.create_operation(
            op_type="Add",
            inputs=[r_comp, g_comp]
        )
        rgb_p = OnnxConstructor.create_operation(
            op_type="Add",
            inputs=[tot_rg, b_comp]
        )


        m = OnnxConstructor.create_operation(
            op_type="Sub",
            inputs=[
                value,
                OnnxConstructor.create_operation(
                    op_type="Div",
                    inputs=[chroma, two]
                )
            ]
        )

        OnnxConstructor.create_output(m, OnnxConstructor.get_shape(m))

        return OnnxConstructor.create_operation(
            op_type="Add",
            inputs=[rgb_p, m]
        )


# endregion


def color_transformation_generator(
        original_space: Type[spaces.ColorType],
        destination_space: Type[spaces.ColorType],
        matrix: np.ndarray) -> Callable[[str], str]:
    """
    Creates a function to generate onnx nodes from a transformation matrix.
    """

    def mtf(input_name: str) -> str:
        next_input = input_name

        # Remove original bit depth
        if original_space.bit_depth is not None:
            original_space_factor = OnnxConstructor.create_constant(
                value=np.asarray([2 ** original_space.bit_depth - 1])
            )
            next_input = OnnxConstructor.create_operation(
                op_type="Div",
                inputs=[next_input, original_space_factor]
            )

        # Remove original gamma
        if original_space.gamma is not None:
            gamma = OnnxConstructor.create_constant(
                value=np.asarray([original_space.gamma])
            )
            next_input = OnnxConstructor.create_operation(
                op_type="Pow",
                inputs=[next_input, gamma]
            )

        # Apply matrix transformation
        matrix_constant = OnnxConstructor.create_constant(matrix)
        next_input = OnnxConstructor.create_operation(
            op_type="MatMul",
            inputs=[next_input, matrix_constant]
        )

        # Apply new gamma
        if destination_space.gamma is not None:
            gamma = OnnxConstructor.create_constant(
                value=np.asarray([1 / destination_space.gamma])
            )
            next_input = OnnxConstructor.create_operation(
                op_type="Pow",
                inputs=[next_input, gamma]
            )

        # Apply new bit depth
        if destination_space.bit_depth is not None:
            destination_space_factor = OnnxConstructor.create_constant(
                value=np.asarray([2 ** destination_space.bit_depth - 1])
            )
            next_input = OnnxConstructor.create_operation(
                op_type="Mul",
                inputs=[next_input, destination_space_factor]
            )

        return next_input

    return mtf



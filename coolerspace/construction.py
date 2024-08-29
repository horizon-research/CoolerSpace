from __future__ import annotations

import onnx
import onnx.helper as oh
import numpy as np

from typing import Sequence


class OnnxConstructor:
    """
    Class that creates and stores onnx nodes, inputs, and outputs
    """

    node_map: dict[str, onnx.NodeProto] = dict()
    """
    Created onnx nodes
    """

    constant_map: dict[int, str] = dict()
    """
    Map of constant tensor hashes to onnx node string
    """

    input_map: dict[str, onnx.ValueInfoProto] = dict()
    """
    Created onnx inputs
    """

    output_map: dict[str, onnx.ValueInfoProto] = dict()
    """
    Created onnx outputs
    """

    shape_map: dict[str, list[int]] = dict()
    """
    Map of node name to shape
    """

    @classmethod
    def create_constant(cls,
                        value: np.ndarray,
                        data_type: onnx.TensorProto.DataType = onnx.TensorProto.DOUBLE) -> str:
        """
        Creates and stores an onnx constant node.
        :returns: Corresponding output string.
        """
        # Ensure that we are not adding a duplicate constant
        constant_hash = hash(hash(value.tobytes()) + hash(value.shape))
        if constant_hash in cls.constant_map.keys():
            return cls.constant_map[constant_hash]

        # Create and constant node
        output_name = cls._get_next_output_name()
        constant_node = oh.make_node(
            op_type="Constant",
            inputs=[],
            outputs=[output_name],
            name=output_name + "_node",
            value=oh.make_tensor(
                name=output_name + "_value",
                data_type=data_type,
                dims=value.shape,
                vals=value.flatten()
            )
        )

        # Populate dictionaries
        cls.constant_map[constant_hash] = output_name
        cls.node_map[output_name] = constant_node

        # Shape inference
        cls.infer_shapes()

        return output_name

    @classmethod
    def create_operation(cls,
                         op_type: str,
                         inputs: list[str],
                         **kwargs) -> str:
        """
        Creates an onnx operation node and stores it in the onnx model.
        Returns the corresponding output string.
        """
        # Create operation node
        output_name = cls._get_next_output_name()
        operation_node = oh.make_node(
            op_type=op_type,
            inputs=inputs,
            outputs=[output_name],
            name=output_name + "_node",
            **kwargs
        )

        # Populate dictionaries
        cls.node_map[output_name] = operation_node

        # Shape inference
        cls.infer_shapes()

        return output_name

    @classmethod
    def create_input(cls,
                     name: str,
                     shape: Sequence[int],
                     data_type: onnx.TensorProto.DataType = onnx.TensorProto.DOUBLE) -> bool:
        """
        Creates and stores an input node.
        :returns: False is input name is already taken. True otherwise.
        """
        # Check if name is already in use
        if cls.check_name_already_in_use(name):
            return False

        # Create input node
        input_node = oh.make_tensor_value_info(
            name=name,
            elem_type=data_type,
            shape=shape
        )

        # Populate dictionaries
        cls.input_map[name] = input_node

        return True

    @classmethod
    def create_output(cls,
                      name: str,
                      shape: Sequence[int],
                      data_type: onnx.TensorProto.DataType = onnx.TensorProto.DOUBLE) -> bool:
        """
        Creates and stores an output node.
        :returns: False if output name is already taken. True otherwise.
        """

        # Create output node
        output_node = oh.make_tensor_value_info(
            name=name,
            elem_type=data_type,
            shape=shape
        )

        # Populate dictionaries
        cls.output_map[name] = output_node

        return True

    @classmethod
    def generate_model(cls) -> onnx.ModelProto:
        """
        Generates and checks model.
        """
        # Generate model
        graph = oh.make_graph(
            nodes=list(cls.node_map.values()),
            name="Generated",
            inputs=list(cls.input_map.values()),
            outputs=list(cls.output_map.values())
        )
        model = oh.make_model(graph)

        # Check model
        onnx.checker.check_model(model)
        model = onnx.shape_inference.infer_shapes(model, strict_mode=True)

        return model

    @classmethod
    def infer_shapes(cls):
        """
        Uses onnx's shape inference API to extract expected shapes from the current set of operations
        """
        # Infer shapes
        model = cls.generate_model()
        # If an error is thrown here, our own shape checking is faulty
        model = onnx.shape_inference.infer_shapes(model, strict_mode=True)

        # Populate shape field
        cls.shape_map = {
            vi.name: [item.dim_value for item in vi.type.tensor_type.shape.dim]
            for vi in model.graph.value_info
        }

    @classmethod
    def get_shape(cls, node_name: str):
        """
        Gets the specific shape of a node
        """
        # First check to see if it is an input
        if node_name in cls.input_map.keys():
            return [item.dim_value for item in cls.input_map[node_name].type.tensor_type.shape.dim]
        return cls.shape_map[node_name]

    @classmethod
    def _get_next_output_name(cls) -> str:
        """
        Returns an unused output name
        """
        return "output{}".format(len(cls.node_map))

    @classmethod
    def check_name_already_in_use(cls, name: str) -> bool:
        """
        :returns: True if name is already in use. False otherwise.
        """
        return name in cls.node_map.keys() or name in cls.input_map.keys() or name in cls.output_map.keys()

    @classmethod
    def check_node_exists(cls, name: str) -> bool:
        """
        Checks if the given name belongs to a node, NOT AN INPUT OR OUTPUT
        """

    @classmethod
    def check_is_input(cls, name: str) -> bool:
        """
        Checks if name belongs to input
        """
        return name in cls.input_map.keys()

    @classmethod
    def check_is_output(cls, name: str) -> bool:
        """
        Checks if name belongs to output
        """
        return name in cls.output_map.keys()

    @classmethod
    def check_is_node(cls, name: str) -> bool:
        """
        Checks if name belongs to node
        """
        return name in cls.node_map.keys()


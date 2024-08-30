# CoolerSpace

CoolerSpace is a Python library that provides type checking for programs that manipulate colors.

## Installation
CoolerSpace is [available on PyPI](https://pypi.org/project/coolerspace/)!
You can install CoolerSpace with pip.

```
pip install coolerspace
```

Alternatively, if you wish to build CoolerSpace yourself, please use the following commands:

```
git clone https://github.com/horizon-research/CoolerSpace.git
cd CoolerSpace
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install build
python3 -m build
```

After running these commands, install the generated wheel file. It should be located in the `dist/` folder.

## Usage
Please refer [here](https://github.com/horizon-research/CoolerSpaceBenchmarker/tree/main/programs) for examples of CoolerSpace programs.

### Import
Please note that the code below assumes that you have imported `coolerspace as cs`.

```
import coolerspace as cs
```

### Inputs
In order to specify a user input value, use the following syntax:

```
x = cs.create_input(input_variable_name,shape, type)
```

Here, `input_variable_name` is the name of the variable used when running the ONNX file.
`shape` is the shape of the input data. For example, a full HD picture would have the shape `[1080, 1920]`.
`type` is the type of the input variable. An example of a valid type is `cs.XYZ`.
Valid types are enumerated in the `spaces.py` file of coolerspace.
Please refer to our paper for an exhaustive list.

### Initializing CoolerSpace objects
Besides using  `cs.create_input`, coolerspace objects can be created via the following constructor

```
cs.sRGB([0, 0, 0])
```

The above line of code creates an sRGB object. `cs.sRGB` can be substituted for other CoolerSpace types.

### Casting CoolerSpace objects
CoolerSpace objects can be cast between different types.

```
cs.sRGB(xyz)
```

The above line of code converts a value of the `cs.XYZ` type to a value of the `cs.sRGB` type. 

### Arithmetic operations
Arithmetic operations can be performed on CoolerSpace objects.
The list of valid arithmetic operations is detailed in our paper.

```
xyz1: cs.XYZ = ...
xyz2: cs.XYZ = ...
xyz3 = xyz1 * 0.5 + xyz2 * 0.5
```

### Casting

### Outputs
Outputs can be created with the following syntax:

```
cs.create_output(cs_object)
```

`cs_object` is a CoolerSpace object you would like to output when running the ONNX program.

### Compilation
Use the `cs.compile` command to specify what path you would like to compile the ONNX file to:

```
cs.compile(path)
```

## Related Repositories
CoolerSpace's output ONNX files can be optimized using equality saturation.
Our optimization tool is stored in a separate GitHub repository, found [here](https://github.com/horizon-research/onneggs).
We also have a benchmarking suite for CoolerSpace, found [here](https://github.com/horizon-research/CoolerSpaceBenchmarker).
The benchmarking suite was used to gather the data found in the paper.
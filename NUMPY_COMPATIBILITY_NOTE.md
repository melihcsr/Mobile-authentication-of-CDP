# NumPy Compatibility Note

## Issue

The project originally required NumPy 1.18.1 as specified in `requirements.txt`, but was encountering errors with NumPy 2.0.2 which is what was installed in the Python environment:

```
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.0.2 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
```

The error occurred when trying to import `matplotlib.pyplot` which depends on NumPy.

## Solution

Downgraded NumPy to a compatible version. NumPy 1.18.1 failed to compile with Python 3.9.18, but NumPy 1.24.4 worked successfully:

```bash
pip install numpy==1.24.4
```

This version allows matplotlib to work properly while maintaining compatibility with the rest of the codebase.

## Note

If you encounter NumPy-related errors while running the code, try using NumPy 1.24.4 which is compatible with both Python 3.9 and the requirements for this project.

You may also need to consider other package dependencies:

```
tensorflow 2.19.0 requires numpy<2.2.0,>=1.26.0, but numpy 1.24.4 is installed.
```

If you need to use TensorFlow 2.19.0, you may need to use a different NumPy version or consider using a virtual environment with the specific version requirements for this project. 
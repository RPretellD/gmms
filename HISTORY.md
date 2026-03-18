# gmms: Versions

### V0.1.0
- Initial release. 

### V0.1.1
- Minor fixes. 

### V0.2.0
- Solved installation issues associated to Cython. Cython is compiled at the during pip installation (if C/C++ tools are available). 
- Functions that were previously available in Cython only are now also available in Python. 
- Most functions are loaded by default using the Python implementation, but can also be used in Cython when desired. 
- Other code changes to accomodate the above updates. 

### V1.0.0
- Included implementations for Ry0 distance. 
- Removed unnecessary variables and variable declarations from some functions. 
- Other minor edits. 

### V1.0.1
- Modified the backend selection to facilitate going from Python (default) to Cython and back to Python. This modification also allows for the backend to be selected from [gmKriger](https://github.com/RPretellD/gmKriger).

- .cu files must not have - in them, use underscores

Build:
```
cmake -B build -DARCH=80
cmake --build build
```

nvbench requires
CMake > 3.30.4

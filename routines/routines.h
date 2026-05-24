#pragma once
#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cuda_runtime.h>
#include <time.h>
#include <unordered_map>
#include <stdexcept>
#include "enums.h"

#define LIB_HAS(name) LIB_HAS_##name

void reduction_handler(RunMode runMode, int argc, char * argv[]);
void scan_handler(RunMode runMode, int argc, char * argv[]);


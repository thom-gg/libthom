#pragma once
#include <stdio.h>
#include "enums.h"

#define LIB_HAS(name) LIB_HAS_##name

void reduction_handler(RunMode runMode, int argc, char * argv[]);
void scan_handler(RunMode runMode, int argc, char * argv[]);


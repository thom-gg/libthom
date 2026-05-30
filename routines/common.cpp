#include "routines.h"


float getRdmFloat(float lowerBound, float upperBound) {
    return lowerBound + ((float)rand() / RAND_MAX) * (upperBound-lowerBound);
}
#include "routines.h"


double getRdmDouble(double lowerBound, double upperBound) {
    return lowerBound + ((float)rand() / RAND_MAX) * (upperBound-lowerBound);
}
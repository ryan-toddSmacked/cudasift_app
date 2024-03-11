#pragma once
#ifndef __GEOMFUNCS_HPP__
#define __GEOMFUNCS_HPP__


#include "cudaSift.h"

int ImproveHomography(SiftData &data, float *homography, int numLoops, float minScore, float maxAmbiguity, float thresh);


#endif // __GEOMFUNCS_HPP__

#ifndef VOLUME_UTILS_H
#define VOLUME_UTILS_H
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Ray.h>
#include <Core/Geometry/BBox.h>

namespace Kurt {
using namespace SCIRun;

bool isPowerOf2( int range);

int nextPowerOf2( int range );

int largestPowerOf2( int range );

double intersectParam(const Vector& N, const Point& P, const Ray& R);

void sortParameters( double *t, int len_t );
} // End namespace Kurt


#endif

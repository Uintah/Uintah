#ifndef VOLUME_UTILS_H
#define VOLUME_UTILS_H
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Ray.h>
#include <SCICore/Geometry/BBox.h>

namespace Kurt {
namespace GeomSpace {

using namespace SCICore::Geometry;

bool isPowerOf2( int range);

int nextPowerOf2( int range );

int largestPowerOf2( int range );

double intersectParam(const Vector& N, const Point& P, const Ray& R);

void sortParameters( double *t, int len_t );

} // end GeomSpace
} // end Kurt

#endif

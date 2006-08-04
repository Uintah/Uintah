#ifndef GEOPROBEREADER_H
#define GEOPROBEREADER_H 1

#include <Packages/rtrt/Core/GpVolHdr.h>
#include <Core/Geometry/Point.h>
#include <Packages/rtrt/Core/Array3.h>

using SCIRun::Point;
using rtrt::Array3;

int read_geoprobe(const char *fname, int &nx, int &ny, int &nz,
		  Point &min, Point &max, 
		  unsigned char &datamin,
		  unsigned char &datamax, 
		  Array3<unsigned char> &data);


#endif


#include "SubRegion.h"
using SCICore::Geometry::Point;

SubRegion::SubRegion(const Point& lower, const Point& upper,
		     int sx, int sy, int sz,
		     int ex, int ey, int ez)
    : lower(lower), upper(upper),
      sx(sx), sy(sy), sz(sz),
      ex(ex), ey(ey), ez(ez)
{
}

SubRegion::SubRegion(const SubRegion& copy)
    : lower(copy.lower), upper(copy.upper),
      sx(copy.sx), sy(copy.sy), sz(copy.sz),
      ex(copy.ex), ey(copy.ey), ez(copy.ez)
{
}

SubRegion::~SubRegion()
{
}


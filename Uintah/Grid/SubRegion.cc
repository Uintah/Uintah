/* REFERENCED */
static char *id="@(#) $Id$";

#include "SubRegion.h"

using namespace Uintah;
using SCICore::Geometry::Point;

SubRegion::SubRegion(const Point& lower, const Point& upper,
		     int sx, int sy, int sz,
		     int ex, int ey, int ez)
    : d_lower(lower), d_upper(upper),
      d_sx(sx), d_sy(sy), d_sz(sz),
      d_ex(ex), d_ey(ey), d_ez(ez)
{
}

SubRegion::SubRegion(const SubRegion& copy)
    : d_lower(copy.d_lower), d_upper(copy.d_upper),
      d_sx(copy.d_sx), d_sy(copy.d_sy), d_sz(copy.d_sz),
      d_ex(copy.d_ex), d_ey(copy.d_ey), d_ez(copy.d_ez)
{
}

SubRegion::~SubRegion()
{
}

//
// $Log$
// Revision 1.3  2000/04/26 06:48:58  sparker
// Streamlined namespaces
//
// Revision 1.2  2000/03/16 22:08:01  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

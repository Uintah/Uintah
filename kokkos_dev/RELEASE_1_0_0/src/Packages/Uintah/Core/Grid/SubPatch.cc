
#include <Packages/Uintah/Core/Grid/SubPatch.h>

using namespace Uintah;
using namespace SCIRun;

SubPatch::SubPatch(const Point& lower, const Point& upper,
		     int sx, int sy, int sz,
		     int ex, int ey, int ez)
    : d_lower(lower), d_upper(upper),
      d_sx(sx), d_sy(sy), d_sz(sz),
      d_ex(ex), d_ey(ey), d_ez(ez)
{
}

SubPatch::SubPatch(const SubPatch& copy)
    : d_lower(copy.d_lower), d_upper(copy.d_upper),
      d_sx(copy.d_sx), d_sy(copy.d_sy), d_sz(copy.d_sz),
      d_ex(copy.d_ex), d_ey(copy.d_ey), d_ez(copy.d_ez)
{
}

SubPatch::~SubPatch()
{
}


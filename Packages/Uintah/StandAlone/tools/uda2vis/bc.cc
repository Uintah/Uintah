
#include <Packages/Uintah/StandAlone/tools/uda2vis/bc.h>

using namespace SCIRun;

bool
is_periodic_bcs( IntVector cellir, IntVector ir )
{
  if( cellir.x() == ir.x() ||
      cellir.y() == ir.y() ||
      cellir.z() == ir.z() )
    return true;
  else
    return false;
}

void
get_periodic_bcs_range( IntVector cellmax, IntVector datamax,
                        IntVector range, IntVector& newrange )
{
  if( cellmax.x() == datamax.x())
    newrange.x( range.x() + 1 );
  else
    newrange.x( range.x() );
  if( cellmax.y() == datamax.y())
    newrange.y( range.y() + 1 );
  else
    newrange.y( range.y() );
  if( cellmax.z() == datamax.z())
    newrange.z( range.z() + 1 );
  else
    newrange.z( range.z() );
}


#include "Analyze.h"

#include <Uintah/Grid/Grid.h>
#include <Uintah/Grid/SimulationState.h>
#include <Uintah/Interface/DataWarehouse.h>

namespace Uintah {

Analyze::Analyze()
: UintahParallelPort()
{
}

Analyze::~Analyze()
{
}
      
void Analyze::setup( const Grid& grid,
                     const SimulationState& sharedState,
                     DataWarehouseP dw)
{
  d_grid = &grid;
  d_sharedState = &sharedState;
  d_dw = dw;
}

} // end namespace Uintah

//
// $Log$
// Revision 1.1  2000/07/17 23:37:35  tan
// Added Analyze interface that will be especially useful for debugging
// on scitific results.
//

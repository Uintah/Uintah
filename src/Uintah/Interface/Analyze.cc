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

} // end namespace Uintah

//
// $Log$
// Revision 1.2  2000/09/04 00:38:03  tan
// Modified Analyze interface for scientific debugging under both
// sigle processor and mpi environment.
//
// Revision 1.1  2000/07/17 23:37:35  tan
// Added Analyze interface that will be especially useful for debugging
// on scitific results.
//


#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Core/Geometry/Vector.h>

#include <iostream>

using namespace Uintah;
using namespace SCIRun;

using std::cerr;

DataWarehouse::DataWarehouse(const ProcessorGroup* myworld,
			     Scheduler* scheduler,
			     int generation)
  : d_myworld(myworld), d_scheduler(scheduler), d_generation(generation)
{
}

DataWarehouse::~DataWarehouse()
{
}

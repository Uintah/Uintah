
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Core/Geometry/Vector.h>

#include <iostream>

using namespace Uintah;
using namespace SCIRun;

using std::cerr;

DataWarehouse::DataWarehouse(const ProcessorGroup* myworld, 
			     int generation)
  : d_myworld(myworld), d_generation( generation )
{
}

DataWarehouse::~DataWarehouse()
{
}


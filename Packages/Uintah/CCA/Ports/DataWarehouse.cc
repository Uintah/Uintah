
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Core/Geometry/Vector.h>

#include <iostream>

using namespace Uintah;
using namespace SCIRun;

using std::cerr;

DataWarehouse::DataWarehouse(const ProcessorGroup* myworld, 
			     int generation, 
			     DataWarehouseP& parent_dw) :
  d_myworld(myworld),
  d_generation( generation ), d_parent(parent_dw)
{
}

DataWarehouse::~DataWarehouse()
{
}

DataWarehouseP
DataWarehouse::getTop() const{
  DataWarehouseP parent = d_parent;
  //while (parent->d_parent) {
   // parent = parent->d_parent;
  //}
  return parent;
}



#include "DataWarehouse.h"
#include <SCICore/Geometry/Vector.h>
#include <iostream>
using std::cerr;
using SCICore::Geometry::Vector;

DataWarehouse::DataWarehouse()
{
}

DataWarehouse::~DataWarehouse()
{
}

#if 0
void DataWarehouse::get(double& value, const std::string& name) const
{
    value=.45;
    cerr << "DataWarehouse::get not done: " << name << "\n";
}

void DataWarehouse::get(CCVariable<Vector>&, const std::string& name,
			const Region*)
{
    cerr << "DataWarehouse::get not done: " << name << "\n";
}
#endif

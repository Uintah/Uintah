/* REFERENCED */
static char *id="@(#) $Id$";

#include "DataWarehouse.h"
#include <SCICore/Geometry/Vector.h>
#include <iostream>

namespace Uintah {
namespace Interface {

using std::cerr;
using SCICore::Geometry::Vector;

DataWarehouse::DataWarehouse()
{
}

DataWarehouse::~DataWarehouse()
{
}

#if 0
void
DataWarehouse::get(double& value, const std::string& name) const
{
    value=.45;
    cerr << "DataWarehouse::get not done: " << name << "\n";
}

void
DataWarehouse::get(CCVariable<Vector>&, const std::string& name,
		   const Region*)
{
    cerr << "DataWarehouse::get not done: " << name << "\n";
}
#endif

} // end namespace Interface
} // end namespace Uintah

//
// $Log$
// Revision 1.2  2000/03/16 22:08:22  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

/* REFERENCED */
static char *id="@(#) $Id$";

#include "DataWarehouse.h"
#include <SCICore/Geometry/Vector.h>
#include <iostream>

using namespace Uintah;
using std::cerr;
using SCICore::Geometry::Vector;

DataWarehouse::DataWarehouse( int MpiRank, int MpiProcesses, int generation ) :
  d_MpiRank( MpiRank ), 
  d_MpiProcesses( MpiProcesses ), 
  d_generation( generation )
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
		   const Patch*)
{
    cerr << "DataWarehouse::get not done: " << name << "\n";
}
#endif

//
// $Log$
// Revision 1.6  2000/05/30 20:19:40  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.5  2000/05/11 20:10:22  dav
// adding MPI stuff.  The biggest change is that old_dws cannot be const and so a large number of declarations had to change.
//
// Revision 1.4  2000/04/26 06:49:10  sparker
// Streamlined namespaces
//
// Revision 1.3  2000/04/19 21:20:04  dav
// more MPI stuff
//
// Revision 1.2  2000/03/16 22:08:22  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

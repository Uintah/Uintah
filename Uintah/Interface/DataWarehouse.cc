/* REFERENCED */
static char *id="@(#) $Id$";

#include "DataWarehouse.h"
#include <SCICore/Geometry/Vector.h>
#include <iostream>

using namespace Uintah;
using std::cerr;
using SCICore::Geometry::Vector;

DataWarehouse::DataWarehouse(const ProcessorGroup* myworld, int generation, 
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
  while (parent->d_parent) {
    parent = parent->d_parent;
  }
  return parent;
}

//
// $Log$
// Revision 1.8  2000/07/28 03:01:07  rawat
// modified createDatawarehouse and added getTop function
//
// Revision 1.7  2000/06/17 07:06:45  sparker
// Changed ProcessorContext to ProcessorGroup
//
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

/* REFERENCED */
static char *id="@(#) $Id$";

#include "Output.h"
#include <iostream>

namespace Uintah {
namespace Interface {

using std::cerr;

Output::Output()
{
}

Output::~Output()
{
}

void
Output::finalizeTimestep(double t, double, const LevelP&,
			 SchedulerP&, const DataWarehouseP&)
{
    //cerr << "Finalizing time step: t=" << t << '\n';
}

} // end namespace Interface
} // end namespace Uintah

//
// $Log$
// Revision 1.2  2000/03/16 22:08:23  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

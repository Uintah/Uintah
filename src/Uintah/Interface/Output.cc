/* REFERENCED */
static char *id="@(#) $Id$";

#include "Output.h"
#include <iostream>

using namespace Uintah;
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

//
// $Log$
// Revision 1.4  2000/04/26 06:49:11  sparker
// Streamlined namespaces
//
// Revision 1.3  2000/03/17 09:30:03  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.2  2000/03/16 22:08:23  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

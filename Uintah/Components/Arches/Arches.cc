/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/Arches/Arches.h>
#include <SCICore/Util/NotFinished.h>

namespace Uintah {
namespace Components {

Arches::Arches()
{
}

Arches::~Arches()
{
}

void Arches::problemSetup(const ProblemSpecP& params, GridP& grid,
		  DataWarehouseP&)
{
    NOT_FINISHED("Arches::problemSetup");
}

void Arches::computeStableTimestep(const LevelP& level,
				   SchedulerP&, DataWarehouseP&)
{
    NOT_FINISHED("Arches::problemSetup");
}

void Arches::timeStep(double t, double dt,
		      const LevelP& level, SchedulerP&,
		      const DataWarehouseP&, DataWarehouseP&)
{
    NOT_FINISHED("Arches::problemSetup");
}

} // end namespace Components
} // end namespace Uintah

//
// $Log$
// Revision 1.3  2000/03/17 09:29:26  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.2  2000/03/16 22:26:13  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

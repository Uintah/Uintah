
#include <Uintah/Components/Arches/Arches.h>
#include <SCICore/Util/NotFinished.h>

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

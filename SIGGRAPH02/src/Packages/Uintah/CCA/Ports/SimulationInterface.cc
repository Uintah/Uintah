
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>

using namespace Uintah;

SimulationInterface::SimulationInterface()
{
}

SimulationInterface::~SimulationInterface()
{
}

void SimulationInterface::scheduleRefine(/* const */ LevelP& coarseLevel, 
		    /* const */ LevelP& fineLevel, 
		    SchedulerP& scheduler)
{
}

void SimulationInterface::scheduleRefineInterface(/* const */ LevelP& coarseLevel, 
			     /* const */ LevelP& fineLevel, 
			     SchedulerP& scheduler)
{
}

void SimulationInterface::scheduleCoarsen(/* const */ LevelP& coarseLevel, 
		     /* const */ LevelP& fineLevel, 
		     SchedulerP& scheduler)
{
}

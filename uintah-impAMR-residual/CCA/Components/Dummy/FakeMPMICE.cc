#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>

namespace Uintah {

class MPMICE : public UintahParallelComponent, public SimulationInterface {

public:

      MPMICE(const ProcessorGroup* myworld);

      virtual ~MPMICE();

      virtual void problemSetup(const ProblemSpecP& params, 
				GridP& grid,
				SimulationStateP&);

      virtual void scheduleInitialize(const LevelP& level,
				      SchedulerP&);
	 
      virtual void sched_paramInit(const LevelP& level,
				   SchedulerP&,
				   DataWarehouseP& old_dw,
				   DataWarehouseP& new_dw);
      
      virtual void scheduleComputeStableTimestep(const LevelP& level,
						 SchedulerP&);

      virtual void scheduleTimeAdvance(double t, double dt,
				       const LevelP& level, 
				       SchedulerP&,
				       DataWarehouseP&, 
				       DataWarehouseP&);

}; // end class MPMICE

} // End namespace Uintah

using namespace Uintah;

MPMICE::MPMICE( const ProcessorGroup* myworld ) :
  UintahParallelComponent(myworld)
{
}

MPMICE::~MPMICE()
{
}

void 
MPMICE::problemSetup(const ProblemSpecP& params, 
		     GridP&,
		     SimulationStateP& sharedState)
{
  printf("-----------------------------------\n");
  printf(" Y O U ' R E   R U N N I N G   F A K E M P M I C E \n");
  printf("  The last person to check in Uintah/StandAlone/sub.mk \n");
  printf("  probably checked it in after it had been tweaked to compile \n");
  printf("  fakeMPM_ICE. \n");
  printf("-----------------------------------\n");
  exit(1);
}

void 
MPMICE::sched_paramInit(const LevelP& level,
			SchedulerP& sched,
			DataWarehouseP& old_dw,
			DataWarehouseP& new_dw)
{
}

// ****************************************************************************
// Schedule initialization
// ****************************************************************************
void 
MPMICE::scheduleInitialize(const LevelP& level,
			   SchedulerP& sched)
{
}

// ****************************************************************************
// schedule computation of stable time step
// ****************************************************************************
void 
MPMICE::scheduleComputeStableTimestep(const LevelP&,
				      SchedulerP&)
{
}

// ****************************************************************************
// Schedule time advance
// ****************************************************************************
void 
MPMICE::scheduleTimeAdvance(double time, double dt,
			    const LevelP& level, 
			    SchedulerP& sched,
			    DataWarehouseP& old_dw, 
			    DataWarehouseP& new_dw)
{
}

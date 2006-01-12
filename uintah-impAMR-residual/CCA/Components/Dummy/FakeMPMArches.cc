#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>

namespace Uintah {

class MPMArches : public UintahParallelComponent, public SimulationInterface {

public:

      MPMArches(const ProcessorGroup* myworld);

      virtual ~MPMArches();

      virtual void problemSetup(const ProblemSpecP& params, 
				GridP& grid,
				SimulationStateP&);

      virtual void scheduleInitialize(const LevelP& level,
				      SchedulerP&,
				      DataWarehouseP&);
	 
      virtual void sched_paramInit(const LevelP& level,
				   SchedulerP&,
				   DataWarehouseP& old_dw,
				   DataWarehouseP& new_dw);
      
      virtual void scheduleComputeStableTimestep(const LevelP& level,
						 SchedulerP&,
						 DataWarehouseP&);

      virtual void scheduleTimeAdvance(double t, double dt,
				       const LevelP& level, 
				       SchedulerP&,
				       DataWarehouseP&, 
				       DataWarehouseP&);

}; // end class MPMArches

} // End namespace Uintah

using namespace Uintah;

MPMArches::MPMArches(const ProcessorGroup* myworld) :
  UintahParallelComponent(myworld)
{
}

MPMArches::~MPMArches()
{
}

void 
MPMArches::problemSetup(const ProblemSpecP& params, 
		     GridP&,
		     SimulationStateP& sharedState)
{
  printf("-----------------------------------\n");
  printf(" Y O U ' R E   R U N N I N G   F A K E M P M A R C H E S \n");
  printf("  The last person to check in Uintah/StandAlone/sub.mk \n");
  printf("  probably checked it in after it had been tweaked to compile \n");
  printf("  fakeMPMArches. \n");
  printf("-----------------------------------\n");
  exit(1);
}

void 
MPMArches::sched_paramInit(const LevelP& level,
			SchedulerP& sched,
			DataWarehouseP& old_dw,
			DataWarehouseP& new_dw)
{
}

// ****************************************************************************
// Schedule initialization
// ****************************************************************************
void 
MPMArches::scheduleInitialize(const LevelP& level,
			   SchedulerP& sched,
			   DataWarehouseP& dw)
{
}

// ****************************************************************************
// schedule computation of stable time step
// ****************************************************************************
void 
MPMArches::scheduleComputeStableTimestep(const LevelP&,
				      SchedulerP&,
				      DataWarehouseP& dw)
{
}

// ****************************************************************************
// Schedule time advance
// ****************************************************************************
void 
MPMArches::scheduleTimeAdvance(double time, double dt,
			    const LevelP& level, 
			    SchedulerP& sched,
			    DataWarehouseP& old_dw, 
			    DataWarehouseP& new_dw)
{
}


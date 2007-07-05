#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
namespace Uintah {

class AMRICE : public UintahParallelComponent, public SimulationInterface {

public:

      AMRICE(const ProcessorGroup* myworld);

      virtual ~AMRICE();

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

}; // end class AMRICE

} // End namespace Uintah

using namespace Uintah;

AMRICE::AMRICE(const ProcessorGroup* myworld) :
  UintahParallelComponent(myworld)
{
}

AMRICE::~AMRICE()
{
}

void 
AMRICE::problemSetup(const ProblemSpecP& params, 
		     GridP&,
		     SimulationStateP& sharedState)
{
  printf("-----------------------------------\n");
  printf(" Y O U ' R E   R U N N I N G   F A K E   A M R I C E \n");
  printf("  The last person to check in Uintah/StandAlone/sub.mk \n");
  printf("  probably checked it in after it had been tweaked to compile \n");
  printf("  fakeAMRICE. \n");
  printf("-----------------------------------\n");
  exit(1);
}

void 
AMRICE::sched_paramInit(const LevelP& level,
			SchedulerP& sched,
			DataWarehouseP& old_dw,
			DataWarehouseP& new_dw)
{
}

// ****************************************************************************
// Schedule initialization
// ****************************************************************************
void 
AMRICE::scheduleInitialize(const LevelP& level,
			   SchedulerP& sched)
{
}

// ****************************************************************************
// schedule computation of stable time step
// ****************************************************************************
void 
AMRICE::scheduleComputeStableTimestep(const LevelP&,
				      SchedulerP&)
{
}

// ****************************************************************************
// Schedule time advance
// ****************************************************************************
void 
AMRICE::scheduleTimeAdvance(double time, double dt,
			    const LevelP& level, 
			    SchedulerP& sched,
			    DataWarehouseP& old_dw, 
			    DataWarehouseP& new_dw)
{
}

#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
namespace Uintah {

class Arches : public UintahParallelComponent, public SimulationInterface {

public:

      Arches(const ProcessorGroup* myworld);

      virtual ~Arches();

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

}; // end class Arches

} // End namespace Uintah

using namespace Uintah;

Arches::Arches(const ProcessorGroup* myworld) :
  UintahParallelComponent(myworld)
{
}

Arches::~Arches()
{
}

void 
Arches::problemSetup(const ProblemSpecP& params, 
		     GridP&,
		     SimulationStateP& sharedState)
{
  printf("-----------------------------------\n");
  printf(" Y O U ' R E   R U N N I N G   F A K E   A R C H E S \n");
  printf("  The last person to check in Uintah/StandAlone/sub.mk \n");
  printf("  probably checked it in after it had been tweaked to compile \n");
  printf("  fakeArches.  If you want the real arches then do the following \n");
  printf("  a) add the following to src/Packages\n\n");
  printf("     Packages/Uintah/CCA/Components/Arches \\ \n");
  printf("     Packages/Uintah/CCA/Components/MPMArches \\ \n\n");
  printf("     below Packages/Uintah/CCA/Components/Examples  \n\n");
  printf("  b) Check to see that in src/Packages/Uintah/CCA/Component/sub.mk \n");
  printf("     you have something like \n");
  printf("          $(SRCDIR)/MPMArches \\ \n");
  printf("          $(SRCDIR)/Arches \\ \n");
  printf("          $(SRCDIR)/Arches/fortran \\ \n");
  printf("          $(SRCDIR)/Arches/Mixing \\ \n");
  printf("          $(SRCDIR)/Arches/Radiation \\ \n");
  printf("          $(SRCDIR)/Arches/Radiation/fortran \\ \n\n");
  printf("   c) Bug the person who last touched the sub.mk files --Todd\n");
  printf("-----------------------------------\n");
  exit(1);
}

void 
Arches::sched_paramInit(const LevelP& level,
			SchedulerP& sched,
			DataWarehouseP& old_dw,
			DataWarehouseP& new_dw)
{
}

// ****************************************************************************
// Schedule initialization
// ****************************************************************************
void 
Arches::scheduleInitialize(const LevelP& level,
			   SchedulerP& sched)
{
}

// ****************************************************************************
// schedule computation of stable time step
// ****************************************************************************
void 
Arches::scheduleComputeStableTimestep(const LevelP&,
				      SchedulerP&)
{
}

// ****************************************************************************
// Schedule time advance
// ****************************************************************************
void 
Arches::scheduleTimeAdvance(double time, double dt,
			    const LevelP& level, 
			    SchedulerP& sched,
			    DataWarehouseP& old_dw, 
			    DataWarehouseP& new_dw)
{
}

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



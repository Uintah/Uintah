#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
namespace Uintah {

class Arches : public UintahParallelComponent, public SimulationInterface {

public:

      Arches(const ProcessorGroup* myworld);

      virtual ~Arches();

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
			   SchedulerP& sched,
			   DataWarehouseP& dw)
{
}

// ****************************************************************************
// schedule computation of stable time step
// ****************************************************************************
void 
Arches::scheduleComputeStableTimestep(const LevelP&,
				      SchedulerP&,
				      DataWarehouseP& dw)
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



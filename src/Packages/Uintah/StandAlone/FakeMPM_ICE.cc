#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
namespace Uintah {

class MPM : public UintahParallelComponent, public SimulationInterface {

public:

      MPM(const ProcessorGroup* myworld);

      virtual ~MPM();

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

}; // end class MPM

} // End namespace Uintah

using namespace Uintah;

MPM::MPM(const ProcessorGroup* myworld) :
  UintahParallelComponent(myworld)
{
}

MPM::~MPM()
{
}

void 
MPM::problemSetup(const ProblemSpecP& params, 
		     GridP&,
		     SimulationStateP& sharedState)
{
}

void 
MPM::sched_paramInit(const LevelP& level,
			SchedulerP& sched,
			DataWarehouseP& old_dw,
			DataWarehouseP& new_dw)
{
}

// ****************************************************************************
// Schedule initialization
// ****************************************************************************
void 
MPM::scheduleInitialize(const LevelP& level,
			   SchedulerP& sched)
{
}

// ****************************************************************************
// schedule computation of stable time step
// ****************************************************************************
void 
MPM::scheduleComputeStableTimestep(const LevelP&,
				      SchedulerP&)
{
}

// ****************************************************************************
// Schedule time advance
// ****************************************************************************
void 
MPM::scheduleTimeAdvance(double time, double dt,
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

namespace Uintah {

class ICE : public UintahParallelComponent, public SimulationInterface {

public:

      ICE(const ProcessorGroup* myworld);

      virtual ~ICE();

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

}; // end class ICE

} // End namespace Uintah

using namespace Uintah;

ICE::ICE(const ProcessorGroup* myworld) :
  UintahParallelComponent(myworld)
{
}

ICE::~ICE()
{
}

void 
ICE::problemSetup(const ProblemSpecP& params, 
		     GridP&,
		     SimulationStateP& sharedState)
{
}

void 
ICE::sched_paramInit(const LevelP& level,
			SchedulerP& sched,
			DataWarehouseP& old_dw,
			DataWarehouseP& new_dw)
{
}

// ****************************************************************************
// Schedule initialization
// ****************************************************************************
void 
ICE::scheduleInitialize(const LevelP& level,
			   SchedulerP& sched)
{
}

// ****************************************************************************
// schedule computation of stable time step
// ****************************************************************************
void 
ICE::scheduleComputeStableTimestep(const LevelP&,
				      SchedulerP&)
{
}

// ****************************************************************************
// Schedule time advance
// ****************************************************************************
void 
ICE::scheduleTimeAdvance(double time, double dt,
			    const LevelP& level, 
			    SchedulerP& sched,
			    DataWarehouseP& old_dw, 
			    DataWarehouseP& new_dw)
{
}

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

MPMICE::MPMICE(const ProcessorGroup* myworld) :
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

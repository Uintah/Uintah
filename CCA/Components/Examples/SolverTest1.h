
#ifndef Packages_Uintah_CCA_Components_Examples_SolverTest1_h
#define Packages_Uintah_CCA_Components_Examples_SolverTest1_h

#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <Packages/Uintah/CCA/Ports/SolverInterface.h>

namespace Uintah {
  class SimpleMaterial;
  class ExamplesLabel;

/**************************************

CLASS
   SolverTest1
   
   SolverTest1 simulation

GENERAL INFORMATION

   SolverTest1.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   SolverTest1

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class SolverTest1 : public UintahParallelComponent, public SimulationInterface {
  public:
    SolverTest1(const ProcessorGroup* myworld);
    virtual ~SolverTest1();

    virtual void problemSetup(const ProblemSpecP& params, GridP& grid,
			      SimulationStateP&);
    virtual void scheduleInitialize(const LevelP& level,
				    SchedulerP& sched);
    virtual void scheduleComputeStableTimestep(const LevelP& level,
					       SchedulerP&);
    virtual void scheduleTimeAdvance( const LevelP& level, 
				      SchedulerP&, int step, int nsteps );
  private:
    void initialize(const ProcessorGroup*,
		    const PatchSubset* patches, const MaterialSubset* matls,
		    DataWarehouse* old_dw, DataWarehouse* new_dw);
    void computeStableTimestep(const ProcessorGroup*,
			       const PatchSubset* patches,
			       const MaterialSubset* matls,
			       DataWarehouse* old_dw, DataWarehouse* new_dw);
    void timeAdvance(const ProcessorGroup*,
		     const PatchSubset* patches,
		     const MaterialSubset* matls,
		     DataWarehouse* old_dw, DataWarehouse* new_dw,
		     LevelP, Scheduler*);
    ExamplesLabel* lb_;
    SimulationStateP sharedState_;
    double delt_;
    SimpleMaterial* mymat_;
    SolverInterface* solver;
    SolverParameters* solver_parameters;    
    bool x_laplacian, y_laplacian, z_laplacian;
    
    SolverTest1(const SolverTest1&);
    SolverTest1& operator=(const SolverTest1&);
	 
  };
}

#endif

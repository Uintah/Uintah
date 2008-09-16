
#ifndef Packages_Uintah_CCA_Components_Examples_HeatEquation_h
#define Packages_Uintah_CCA_Components_Examples_HeatEquation_h

#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>

#include <Packages/Uintah/CCA/Components/Examples/uintahshare.h>
namespace Uintah {
  class SimpleMaterial;
  class ExamplesLabel;

/**************************************

CLASS
   HeatEquation
   
   HeatEquation simulation

GENERAL INFORMATION

   HeatEquation.h

   John Schmidt
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2008 SCI Group

KEYWORDS
   HeatEquation

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class UINTAHSHARE HeatEquation : public UintahParallelComponent, public SimulationInterface {
  public:
    HeatEquation(const ProcessorGroup* myworld);
    virtual ~HeatEquation();

    virtual void problemSetup(const ProblemSpecP& params, 
                              const ProblemSpecP& restart_prob_spec, 
                              GridP& grid, SimulationStateP&);
    virtual void scheduleInitialize(const LevelP& level,
				    SchedulerP& sched);
    virtual void scheduleComputeStableTimestep(const LevelP& level,
					       SchedulerP&);
    virtual void scheduleTimeAdvance( const LevelP& level, 
				      SchedulerP&);
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
		     DataWarehouse* old_dw, DataWarehouse* new_dw);

    void iterate(const ProcessorGroup*,
		 const PatchSubset* patches,
		 const MaterialSubset* matls,
		 DataWarehouse* old_dw, DataWarehouse* new_dw);

    const VarLabel* temperature_label;
    const VarLabel* residual_label;
    SimulationStateP sharedState_;
    double delt_;
    double maxresidual_;
    SimpleMaterial* mymat_;

    HeatEquation(const HeatEquation&);
    HeatEquation& operator=(const HeatEquation&);
	 
  };
}

#endif

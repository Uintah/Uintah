
#ifndef Packages_Uintah_CCA_Components_Examples_Wave_h
#define Packages_Uintah_CCA_Components_Examples_Wave_h

#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Task.h>

namespace Uintah {
  class SimpleMaterial;
  class ExamplesLabel;

/**************************************

CLASS
   Wave
   
   Wave simulation

GENERAL INFORMATION

   Wave.h

   Steven Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2002 SCI Group

KEYWORDS
   Wave

DESCRIPTION
  
WARNING
  
****************************************/

  class VarLabel;
  class Wave : public UintahParallelComponent, public SimulationInterface {
  public:
    Wave(const ProcessorGroup* myworld);
    virtual ~Wave();

    virtual void problemSetup(const ProblemSpecP& params, GridP& grid,
			      SimulationStateP&);
    virtual void scheduleInitialize(const LevelP& level,
				    SchedulerP& sched);
    virtual void scheduleComputeStableTimestep(const LevelP& level,
					       SchedulerP&);
    virtual void scheduleTimeAdvance( const LevelP& level, 
				      SchedulerP&, int step, int nsteps );
  protected:
    struct Step {
      Task::WhichDW cur_dw;
      const VarLabel* curphi_label;
      const VarLabel* curpi_label;
      const VarLabel* newphi_label;
      const VarLabel* newpi_label;
      double stepweight;
      double totalweight;
    };


    inline double curl(constCCVariable<double>& phi, const IntVector& c, double sumdx2, Vector inv_dx2) 
    {
      return sumdx2 * phi[c]
        + (phi[c+IntVector(1,0,0)] + phi[c-IntVector(1,0,0)]) * inv_dx2.x()
        + (phi[c+IntVector(0,1,0)] + phi[c-IntVector(0,1,0)]) * inv_dx2.y()
        + (phi[c+IntVector(0,0,1)] + phi[c-IntVector(0,0,1)]) * inv_dx2.z();
    }

    void initialize(const ProcessorGroup*,
		    const PatchSubset* patches, const MaterialSubset* matls,
		    DataWarehouse* old_dw, DataWarehouse* new_dw);
    void computeStableTimestep(const ProcessorGroup*,
			       const PatchSubset* patches,
			       const MaterialSubset* matls,
			       DataWarehouse* old_dw, DataWarehouse* new_dw);
    void timeAdvanceEuler(const ProcessorGroup*,
		     const PatchSubset* patches,
		     const MaterialSubset* matls,
		     DataWarehouse* old_dw, DataWarehouse* new_dw, int step, int nsteps);
    void setupRK4(const ProcessorGroup*,
		     const PatchSubset* patches,
		     const MaterialSubset* matls,
		     DataWarehouse* old_dw, DataWarehouse* new_dw, int step, int nsteps);
    void timeAdvanceRK4(const ProcessorGroup*,
		     const PatchSubset* patches,
		     const MaterialSubset* matls,
                        DataWarehouse* old_dw, DataWarehouse* new_dw, Step* s, int step, int nsteps);
    virtual void addRefineDependencies( Task* /*task*/, const VarLabel* /*label*/,
                                        int /*step*/, int /*nsteps*/ ) {}
    virtual void refineFaces(const Patch* /*finePatch*/, const Level* /*fineLevel*/, const Level* /*coarseLevel*/, 
                             CCVariable<double>& /*finevar*/, const VarLabel* /*label*/, int /*step*/, int /*nsteps*/,
                             int /*matl*/, DataWarehouse* /*coarse_old_dw*/, DataWarehouse* /*coarse_new_dw*/) {}

    const VarLabel* phi_label;
    const VarLabel* pi_label;
    double r0;
    string initial_condition;
    string integration;
    SimulationStateP sharedState_;
    SimpleMaterial* mymat_;
    Step rk4steps[4];

    Wave(const Wave&);
    Wave& operator=(const Wave&);
	 
  };
}

#endif

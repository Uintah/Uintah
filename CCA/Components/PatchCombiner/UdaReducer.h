#ifndef UINTAH_HOMEBREW_Component_UdaReducer_H
#define UINTAH_HOMEBREW_Component_UdaReducer_H

#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/Core/DataArchive/DataArchive.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>

#include <Packages/Uintah/CCA/Components/PatchCombiner/share.h>

#include <vector>

namespace Uintah {
  class LoadBalancer;

/**************************************

CLASS
   SimulationInterface
   
   Short description...

GENERAL INFORMATION

   SimulationInterface.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Simulation_Interface

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class SCISHARE UdaReducer : public SimulationInterface, public UintahParallelComponent {
   public:
     UdaReducer(const ProcessorGroup* myworld, string udaDir);
     
     virtual ~UdaReducer();
     
     //////////
     // Insert Documentation Here:
     virtual void problemSetup(const ProblemSpecP& params, 
                               const ProblemSpecP& materials_ps, 
                               GridP& grid, SimulationStateP& state);
     
     //////////
     // Insert Documentation Here:
     virtual void scheduleInitialize(const LevelP& level,
				     SchedulerP&);
      //////////
     // Insert Documentation Here:
     virtual void restartInitialize() {}
     
     //////////
     // Insert Documentation Here:
     virtual void scheduleComputeStableTimestep(const LevelP&,
						SchedulerP&) {}
     
     //////////
     // Insert Documentation Here:
     virtual void scheduleTimeAdvance( const LevelP& level, SchedulerP&);
     

     virtual bool needRecompile(double time, double dt,
                                const GridP& grid);
     double getMaxTime();

     GridP getGrid() { return oldGrid_; }
   private:
     UdaReducer(const UdaReducer&);
     UdaReducer& operator=(const UdaReducer&);

     void initialize(const ProcessorGroup*,
			 const PatchSubset* patches,
			 const MaterialSubset* matls,
			 DataWarehouse* /*old_dw*/,
			 DataWarehouse* new_dw);
       
     void readAndSetDelT(const ProcessorGroup*,
			 const PatchSubset* patches,
			 const MaterialSubset* matls,
			 DataWarehouse* /*old_dw*/,
			 DataWarehouse* new_dw);

     void readAndSetVars(const ProcessorGroup*,
			 const PatchSubset* patches,
			 const MaterialSubset* matls,
			 DataWarehouse* /*old_dw*/,
			 DataWarehouse* new_dw);

     string udaDir_;
     DataArchive* dataArchive_;
     std::vector<int> timesteps_;
     std::vector<double> times_;
     int timeIndex_;
     std::vector<int> numMaterials_;
     GridP oldGrid_;
     LoadBalancer* lb;
     VarLabel* delt_label;
     std::vector<VarLabel*> labels_;
     SimulationStateP d_sharedState;
   };
} // End namespace Uintah
   


#endif


#ifndef Packages_Uintah_CCA_Components_Examples_flameSheet_rxn_h
#define Packages_Uintah_CCA_Components_Examples_flameSheet_rxn_h

#include <Packages/Uintah/CCA/Ports/ModelInterface.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>

#include <map>
#include <vector>

namespace Uintah {
  
/**************************************

CLASS
   flameSheet_rxn
   
   flameSheet_rxn simulation

GENERAL INFORMATION

   flameSheet_rxn.h

   SteveParker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   flameSheet_rxn

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/
  class ICELabel;
  class GeometryPiece;
  class flameSheet_rxn : public ModelInterface {
  public:
    flameSheet_rxn(const ProcessorGroup* myworld, ProblemSpecP& params);
    virtual ~flameSheet_rxn();
    
    virtual void problemSetup(GridP& grid, SimulationStateP& sharedState,
			      ModelSetup* setup);
    
    virtual void scheduleInitialize(SchedulerP&,
				    const LevelP& level,
				    const ModelInfo*);

    virtual void restartInitialize() {}
      
    virtual void scheduleComputeStableTimestep(SchedulerP&,
					       const LevelP& level,
					       const ModelInfo*);
      
    virtual void scheduleComputeModelSources(SchedulerP&,
						   const LevelP& level,
						   const ModelInfo*);
                                             
    virtual void scheduleModifyThermoTransportProperties(SchedulerP&,
                                               const LevelP&,
                                               const MaterialSet*);
                                               
   virtual void computeSpecificHeat(CCVariable<double>&,
                                    const Patch*,
                                    DataWarehouse*,
                                    const int);
                                    
   virtual void scheduleErrorEstimate(const LevelP& coarseLevel,
                                      SchedulerP& sched);
                                      
   virtual void scheduleTestConservation(SchedulerP&,
                                         const PatchSet* patches,
                                         const ModelInfo* mi);           

  private:
    ICELabel* lb;
    
    void initialize(const ProcessorGroup*, 
                    const PatchSubset* patches,
		      const MaterialSubset* matls, 
                    DataWarehouse*, 
		      DataWarehouse* new_dw);
                     
    void testConservation(const ProcessorGroup*, 
                          const PatchSubset* patches,
                          const MaterialSubset*,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw,
                          const ModelInfo* mi);   
   
    void computeModelSources(const ProcessorGroup*, 
                             const PatchSubset* patches,
	                      const MaterialSubset* matls, 
                             DataWarehouse*, 
	                      DataWarehouse* new_dw, 
                             const ModelInfo*);

    flameSheet_rxn(const flameSheet_rxn&);
    flameSheet_rxn& operator=(const flameSheet_rxn&);

    ProblemSpecP params;

    const Material* d_matl;
    MaterialSet* d_matl_set;

    class Region {
    public:
      Region(GeometryPiece* piece, ProblemSpecP&);

      GeometryPiece* piece;
      double initialScalar;
    };

    class Scalar {
    public:
      int index;
      string name;
      VarLabel* scalar_CCLabel;
      VarLabel* scalar_source_CCLabel;
      VarLabel* sum_scalar_fLabel;
      vector<Region*> regions;
    };

    Scalar* d_scalar;
    double d_del_h_comb;
    double d_f_stoic;
    double d_cp;
    double d_T_oxidizer_inf;
    double d_T_fuel_init;
    double d_diffusivity;
    int  d_smear_initialDistribution_knob;
    bool d_test_conservation;
    SimulationStateP d_sharedState;
  };
}

#endif

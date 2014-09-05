
#ifndef Packages_Uintah_CCA_Components_Examples_flameSheet_rxn_h
#define Packages_Uintah_CCA_Components_Examples_flameSheet_rxn_h

#include <Packages/Uintah/CCA/Ports/ModelInterface.h>
#include <Packages/Uintah/Core/Grid/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>

#include <map>
#include <vector>

namespace Uintah {

/**************************************

CLASS
   flameSheet_rxn
   
   flameSheet_rxn simulation

GENERAL INFORMATION

   flameSheet_rxn.h

   Todd Harman
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
      
    virtual void scheduleMassExchange(SchedulerP&,
				      const LevelP& level,
				      const ModelInfo*);
    virtual void scheduleMomentumAndEnergyExchange(SchedulerP&,
						   const LevelP& level,
						   const ModelInfo*);

  private:
    void initialize(const ProcessorGroup*, 
                    const PatchSubset* patches,
		      const MaterialSubset* matls, DataWarehouse*, 
		      DataWarehouse* new_dw);
                     
    void react(const ProcessorGroup*, 
              const PatchSubset* patches,
	       const MaterialSubset* matls, DataWarehouse*, 
	       DataWarehouse* new_dw, const ModelInfo*);

template <class T> 
   void q_diffusion(CellIterator iter, 
                    IntVector adj_offset,
                    const double diffusivity,
                    const double dx,   
                    const CCVariable<double>& q_CC,
                    T& q_FC);
                    
   void computeQ_diffusion_FC(DataWarehouse* new_dw,
                              const Patch* patch,  
                              const CCVariable<double>& q_CC,
                              const double diffusivity,
                              SFCXVariable<double>& q_X_FC,
                              SFCYVariable<double>& q_Y_FC,
                              SFCZVariable<double>& q_Z_FC);

    flameSheet_rxn(const flameSheet_rxn&);
    flameSheet_rxn& operator=(const flameSheet_rxn&);

    ProblemSpecP params;

    const Material* matl;
    MaterialSet* mymatls;

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
      vector<Region*> regions;
    };

    vector<Scalar*> scalars;
    map<string, Scalar*> names;
    double d_del_h_comb;
    double d_f_stoic;
    double d_cp;
    double d_T_oxidizer_inf;
    double d_T_fuel_init;
    double d_diffusivity;
    int  d_smear_initialDistribution_knob;
    SimulationStateP sharedState;
  };
}

#endif


#ifndef Packages_Uintah_CCA_Components_Examples_flameSheet_rxn_h
#define Packages_Uintah_CCA_Components_Examples_flameSheet_rxn_h

#include <Packages/Uintah/CCA/Ports/ModelInterface.h>
#include <Packages/Uintah/Core/Grid/ComputeSet.h>
#include <map>
#include <vector>

namespace Uintah {

/**************************************

CLASS
   flameSheet_rxn
   
   flameSheet_rxn simulation

GENERAL INFORMATION

   flameSheet_rxn.h

   Steven G. Parker
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
    
    //////////
    // Insert Documentation Here:
    virtual void problemSetup(GridP& grid, SimulationStateP& sharedState,
			      ModelSetup* setup);
    
    //////////
    // Insert Documentation Here:
    virtual void scheduleInitialize(SchedulerP&,
				    const LevelP& level,
				    const ModelInfo*);

    //////////
    // Insert Documentation Here:
    virtual void restartInitialize() {}
      
    //////////
    // Insert Documentation Here:
    virtual void scheduleComputeStableTimestep(SchedulerP&,
					       const LevelP& level,
					       const ModelInfo*);
      
    //////////
    // Insert Documentation Here:
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

    double d_cv;
    double d_gamma;
    double d_cp;
    SimulationStateP sharedState;
  };
}

#endif


#ifndef Packages_Uintah_CCA_Components_Examples_Mixing_h
#define Packages_Uintah_CCA_Components_Examples_Mixing_h

#include <Packages/Uintah/CCA/Ports/ModelInterface.h>
#include <Packages/Uintah/Core/Grid/ComputeSet.h>

namespace Uintah {

/**************************************

CLASS
   Mixing
   
   Mixing simulation

GENERAL INFORMATION

   Mixing.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Mixing

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class GeometryPiece;
  class Mixing : public ModelInterface {
  public:
    Mixing(const ProcessorGroup* myworld, ProblemSpecP& params);
    virtual ~Mixing();
    
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
    void massExchange(const ProcessorGroup*, const PatchSubset* patches,
		      const MaterialSubset* matls, DataWarehouse*, 
		      DataWarehouse* new_dw, const ModelInfo*);
    void initialize(const ProcessorGroup*, const PatchSubset* patches,
		    const MaterialSubset* matls, DataWarehouse*, 
		    DataWarehouse* new_dw);

    Mixing(const Mixing&);
    Mixing& operator=(const Mixing&);

    ProblemSpecP params;

    const Material* matl;
    MaterialSet* mymatls;

    class Region {
    public:
      Region(GeometryPiece* piece, ProblemSpecP&);

      GeometryPiece* piece;
      double initialMassFraction;
    };

    class Stream {
    public:
      int index;
      //MaterialProperties props;
      VarLabel* massFraction_CCLabel;
      vector<Region*> regions;
    };

    vector<Stream*> streams;
  };
}

#endif

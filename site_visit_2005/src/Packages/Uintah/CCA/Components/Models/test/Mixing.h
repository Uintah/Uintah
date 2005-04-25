
#ifndef Packages_Uintah_CCA_Components_Examples_Mixing_h
#define Packages_Uintah_CCA_Components_Examples_Mixing_h

#include <Packages/Uintah/CCA/Ports/ModelInterface.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <Packages/Uintah/CCA/Components/Models/test/MaterialProperties.h>

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
    void initialize(const ProcessorGroup*, 
                    const PatchSubset* patches,
		      const MaterialSubset* matls, 
                    DataWarehouse*, 
		      DataWarehouse* new_dw);
                  
    void computeModelSources(const ProcessorGroup*, 
                             const PatchSubset* patches,
	                      const MaterialSubset* matls, 
                             DataWarehouse*, 
	                      DataWarehouse* new_dw, 
                             const ModelInfo*);

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
      string name;
      MaterialProperties props;
      VarLabel* massFraction_CCLabel;
      VarLabel* massFraction_source_CCLabel;
      vector<Region*> regions;
    };

    class Reaction {
    public:
      int fromStream;
      int toStream;
      double energyRelease;
      double rate;
    };

    vector<Stream*> streams;
    vector<Reaction*> reactions;
  };
}

#endif

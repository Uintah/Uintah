
#ifndef Packages_Uintah_CCA_Components_Examples_Mixing2_h
#define Packages_Uintah_CCA_Components_Examples_Mixing2_h

#include <Packages/Uintah/CCA/Ports/ModelInterface.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <map>
#include <vector>

namespace Cantera {
  class IdealGasMix;
  class Reactor;
}

namespace Uintah {

/**************************************

CLASS
   Mixing2
   
   Mixing2 simulation

GENERAL INFORMATION

   Mixing2.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Mixing2

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class GeometryPiece;
  class Mixing2 : public ModelInterface {
  public:
    Mixing2(const ProcessorGroup* myworld, ProblemSpecP& params);
    virtual ~Mixing2();
    

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

    Mixing2(const Mixing2&);
    Mixing2& operator=(const Mixing2&);

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
      VarLabel* massFraction_CCLabel;
      VarLabel* massFraction_source_CCLabel;
      vector<Region*> regions;
    };

    vector<Stream*> streams;
    map<string, Stream*> names;

    Cantera::IdealGasMix* gas;
    Cantera::Reactor* reactor;
    SimulationStateP sharedState;
  };
}

#endif

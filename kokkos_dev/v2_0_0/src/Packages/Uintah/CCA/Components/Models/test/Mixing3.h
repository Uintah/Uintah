
#ifndef Packages_Uintah_CCA_Components_Examples_Mixing3_h
#define Packages_Uintah_CCA_Components_Examples_Mixing3_h

#include <Packages/Uintah/CCA/Ports/ModelInterface.h>
#include <Packages/Uintah/Core/Grid/ComputeSet.h>
#include <Core/Containers/FastHashTable.h>
#include <map>
#include <vector>

namespace Cantera {
  class IdealGasMix;
  class Reactor;
};

namespace Uintah {

/**************************************

CLASS
   Mixing3
   
   Mixing3 simulation

GENERAL INFORMATION

   Mixing3.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Mixing3

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class M3Key;
  class GeometryPiece;
  class Mixing3 : public ModelInterface {
  public:
    Mixing3(const ProcessorGroup* myworld, ProblemSpecP& params);
    virtual ~Mixing3();
    
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
    void initialize(const ProcessorGroup*, const PatchSubset* patches,
		    const MaterialSubset* matls, DataWarehouse*, 
		    DataWarehouse* new_dw);
    void react(const ProcessorGroup*, const PatchSubset* patches,
	       const MaterialSubset* matls, DataWarehouse*, 
	       DataWarehouse* new_dw, const ModelInfo*);

    Mixing3(const Mixing3&);
    Mixing3& operator=(const Mixing3&);

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
    double d_cv;
    double d_gamma;
    double d_cp;
    SimulationStateP sharedState;

    double dtemp;
    double dpress;
    double dmf;
    double dtfactor;
    SCIRun::FastHashTable<M3Key> table;
    double lookup(int nsp, int idt, int itemp, int ipress, int* imf, double* outmf);
    long nlook;
    long nmiss;
  };
}

#endif

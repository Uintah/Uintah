
#ifndef Packages_Uintah_CCA_Components_Examples_PassiveScalar_h
#define Packages_Uintah_CCA_Components_Examples_PassiveScalar_h
#include <Packages/Uintah/CCA/Ports/ModelInterface.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <map>
#include <vector>

namespace Uintah {
  

/**************************************

CLASS
   PassiveScalar
   
   PassiveScalar simulation

GENERAL INFORMATION

   PassiveScalar.h

   Todd Harman
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   PassiveScalar

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/
  class ICELabel;
  class GeometryPiece;
  class PassiveScalar :public ModelInterface {
  public:
    PassiveScalar(const ProcessorGroup* myworld, ProblemSpecP& params);
    virtual ~PassiveScalar();
    
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
                                                
   void modifyThermoTransportProperties(const ProcessorGroup*, 
                                        const PatchSubset* patches,        
                                        const MaterialSubset*,             
                                        DataWarehouse*,                    
                                        DataWarehouse* new_dw);             
   
    void initialize(const ProcessorGroup*, 
                    const PatchSubset* patches,
		      const MaterialSubset* matls, 
                    DataWarehouse*, 
		      DataWarehouse* new_dw);
                                   
    void computeModelSources(const ProcessorGroup*, 
                             const PatchSubset* patches,
                             const MaterialSubset*,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw,
                             const ModelInfo* mi);
                             
    void testConservation(const ProcessorGroup*, 
                          const PatchSubset* patches,
                          const MaterialSubset*,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw,
                          const ModelInfo* mi);
                                                       
    void errorEstimate(const ProcessorGroup* pg,
    		         const PatchSubset* patches,
			  const MaterialSubset* matl,
			  DataWarehouse* old_dw,
			  DataWarehouse* new_dw,
                       bool initial);

    PassiveScalar(const PassiveScalar&);
    PassiveScalar& operator=(const PassiveScalar&);

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
      // labels for this particular scalar
      VarLabel* scalar_CCLabel;
      VarLabel* scalar_source_CCLabel;
      VarLabel* scalar_gradLabel;
      VarLabel* diffusionCoefLabel;
      
      vector<Region*> regions;
      double diff_coeff;
      double refineCriteria;
      int  initialize_diffusion_knob;
    };
    
    // general labels
    class PassiveScalarLabel {
    public:
      VarLabel* lastProbeDumpTimeLabel;
      VarLabel* sum_scalar_fLabel;
    };
    
    PassiveScalarLabel* Slb;
    Scalar* d_scalar;
    
    SimulationStateP d_sharedState;
    
    //__________________________________
    // global constants
    vector<Vector> d_probePts;
    vector<string> d_probePtsNames;
    bool d_usingProbePts;
    bool d_test_conservation;
    double d_probeFreq;
  };
}

#endif


#ifndef Packages_Uintah_CCA_Components_Examples_TableTest_h
#define Packages_Uintah_CCA_Components_Examples_TableTest_h
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
  class ICELabel;
  class TableInterface;

/**************************************

CLASS
   TableTest
   
   TableTest simulation

GENERAL INFORMATION

   TableTest.h

   Steve Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   TableTest

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class GeometryPiece;
  class TableTest :public ModelInterface {
  public:
    TableTest(const ProcessorGroup* myworld, ProblemSpecP& params);
    virtual ~TableTest();
    
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

    TableTest(const TableTest&);
    TableTest& operator=(const TableTest&);

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
      VarLabel* diffusionCoefLabel;
      
      vector<Region*> regions;
      double f_stoic;
      double diff_coeff;
      int  initialize_diffusion_knob;
    };
    
    // general labels
    class TableTestLabel {
    public:
      VarLabel* lastProbeDumpTimeLabel;
    };
    
    TableTestLabel* Slb;
    Scalar* d_scalar;
    double d_rho_air;
    double d_rho_fuel;
    double d_cv_air;
    double d_cv_fuel;
    double d_R_air;
    double d_R_fuel;
    double d_thermalCond_air;
    double d_thermalCond_fuel;
    double d_viscosity_air;
    double d_viscosity_fuel;
    
    SimulationStateP sharedState;
    Output* dataArchiver;
    vector<Vector> d_probePts;
    vector<string> d_probePtsNames;
    bool d_usingProbePts;
    double d_probeFreq;

    TableInterface* table;
    struct TableValue {
      string name;
      int index;
      VarLabel* label;
    };
    vector<TableValue*> tablevalues;

    int temp_index;
  };
}

#endif

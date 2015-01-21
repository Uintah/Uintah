
#ifndef Packages_Uintah_CCA_Components_Examples_AdiabaticTable_h
#define Packages_Uintah_CCA_Components_Examples_AdiabaticTable_h

#include <CCA/Ports/ModelInterface.h>

#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/VarTypes.h>

#include <vector>

namespace Uintah {
  class ICELabel;
  class TableInterface;

/**************************************

CLASS
   AdiabaticTable
   
   AdiabaticTable simulation

GENERAL INFORMATION

   AdiabaticTable.h

   Todd Harman
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   AdiabaticTable

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class AdiabaticTable :public ModelInterface {
  public:
    AdiabaticTable(const ProcessorGroup* myworld, 
                   ProblemSpecP& params,
                   const bool doAMR);
                   
    virtual ~AdiabaticTable();

    virtual void outputProblemSpec(ProblemSpecP& ps);
    
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

    void computeScaledVariance(const Patch* Patch,
                               DataWarehouse* new_dw,                                 
                               const int indx,                                        
                               constCCVariable<double> f_old,                         
                               vector<constCCVariable<double> >& ind_vars); 
                               
    void  errorEstimate(const ProcessorGroup*,
                           const PatchSubset* patches,
                           const MaterialSubset*,
                           DataWarehouse*,
                           DataWarehouse* new_dw,
                        bool);    

    //__________________________________
    AdiabaticTable(const AdiabaticTable&);
    AdiabaticTable& operator=(const AdiabaticTable&);

    ProblemSpecP params;

    const Material* d_matl;
    MaterialSet* d_matl_set;

    class Region {
    public:
      Region(GeometryPieceP piece, ProblemSpecP&);
      
      void outputProblemSpec(ProblemSpecP&);

      GeometryPieceP piece;
      double initialScalar;
    };

    class Scalar {
    public:
      void outputProblemSpec(ProblemSpecP&);
      int index;
      string name;
      // labels for this particular scalar
      VarLabel* scalar_CCLabel;
      VarLabel* scalar_src_CCLabel;
      VarLabel* mag_grad_scalarLabel;
      VarLabel* diffusionCoeffLabel;
      VarLabel* varianceLabel;
      VarLabel* scaledVarianceLabel;
      VarLabel* sum_scalar_fLabel;
      
      vector<Region*> regions;
      double diff_coeff;
      double refineCriteria;
      bool d_test_conservation;
      bool d_doTableTest;
    };

    double oldProbeDumpTime;
    Scalar* d_scalar;
    // Release is J/kg
    VarLabel* cumulativeEnergyReleased_CCLabel;
    VarLabel* cumulativeEnergyReleased_src_CCLabel;
    
    SimulationStateP d_sharedState;
    Output* dataArchiver;

    TableInterface* table;
    struct TableValue {
      void outputProblemSpec(ProblemSpecP& ps)
      {
        ps->appendElement("tableValue",name);

      };
      string name;
      int index;
      VarLabel* label;
    };
    vector<TableValue*> tablevalues;
    
    //__________________________________
    // global constants
    bool d_doAMR;
    vector<Vector> d_probePts;
    vector<string> d_probePtsNames;
    bool d_usingProbePts;
    double d_probeFreq;
    
    int d_density_index;
    int d_gamma_index;
    int d_cv_index;
    int d_viscosity_index;
    int d_thermalcond_index;
    int d_temp_index;
    int d_MW_index;
    int d_ref_cv_index;
    int d_ref_gamma_index;
    int d_ref_temp_index;

    bool d_useVariance;
    double d_varianceScale;
    double d_varianceMax;
  };

} // end namespace Uintah

#endif


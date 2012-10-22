/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#ifndef Packages_Uintah_CCA_Components_Examples_SimpleRxn_h
#define Packages_Uintah_CCA_Components_Examples_SimpleRxn_h
#include <CCA/Ports/ModelInterface.h>

#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/VarTypes.h>

#include <map>
#include <vector>

namespace Uintah {
  class ICELabel;

/**************************************

CLASS
   SimpleRxn
   
   SimpleRxn simulation

GENERAL INFORMATION

   SimpleRxn.h

   Todd Harman
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   
KEYWORDS
   SimpleRxn

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class SimpleRxn :public ModelInterface {
  public:
    SimpleRxn(const ProcessorGroup* myworld, ProblemSpecP& params);
    virtual ~SimpleRxn();

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
    //__________________________________
    SimpleRxn(const SimpleRxn&);
    SimpleRxn& operator=(const SimpleRxn&);

    ProblemSpecP params;

    const Material* d_matl;
    MaterialSet* d_matl_set;

    class Region {
    public:
      Region(GeometryPieceP piece, ProblemSpecP&);

      GeometryPieceP piece;
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
    class SimpleRxnLabel {
    public:
      VarLabel* lastProbeDumpTimeLabel;
      VarLabel* sum_scalar_fLabel;
    };
    
    SimpleRxnLabel* Slb;
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
    bool d_test_conservation;
    double d_probeFreq;
  };
}

#endif

/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#ifndef Packages_Uintah_CCA_Components_Examples_PassiveScalar_h
#define Packages_Uintah_CCA_Components_Examples_PassiveScalar_h

#include <CCA/Ports/ModelInterface.h>
#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/Grid/Variables/VarTypes.h>

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
  
   
KEYWORDS
   PassiveScalar

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/
  class ICELabel;
  class PassiveScalar :public ModelInterface {
  public:
    PassiveScalar(const ProcessorGroup* myworld, 
                   ProblemSpecP& params,
                  const bool doAMR);
    virtual ~PassiveScalar();

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
    const MaterialSubset* d_matl_sub;

    class Region {
    public:
      Region(GeometryPieceP piece, ProblemSpecP&);

      GeometryPieceP piece;
      double initialScalar;
      bool  sinusoidalInitialize;
      IntVector freq;
      bool  linearInitialize;
      Vector slope;
      bool cubicInitialize;
      Vector direction;
      bool quadraticInitialize;
      Vector coeff;
      bool exponentialInitialize_1D;
      bool exponentialInitialize_2D;
      bool triangularInitialize;
      
      bool  uniformInitialize;
    };

    class Scalar {
    public:
      int index;
      string name;
      // labels for this particular scalar
      VarLabel* scalar_CCLabel;
      VarLabel* scalar_source_CCLabel;
      VarLabel* mag_grad_scalarLabel;
      VarLabel* diffusionCoefLabel;
      
      vector<Region*> regions;
      double diff_coeff;
      double refineCriteria;
      int  initialize_diffusion_knob;
    };
    
    // general labels
    class PassiveScalarLabel {
    public:
      VarLabel* sum_scalar_fLabel;
    };
    
    PassiveScalarLabel* Slb;
    Scalar* d_scalar;
    SimulationStateP d_sharedState;
    
    //__________________________________
    // global constants
    bool d_doAMR;
    bool d_test_conservation;
  };
}

#endif

/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
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


#ifndef Packages_Uintah_CCA_Components_Examples_Mixing2_h
#define Packages_Uintah_CCA_Components_Examples_Mixing2_h

#include <CCA/Components/Models/FluidsBased/FluidsBasedModel.h>

#include <Core/Grid/Variables/ComputeSet.h>
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
  
   
KEYWORDS
   Mixing2

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class ICELabel;

  class GeometryPiece;
  class Mixing2 : public FluidsBasedModel {
  public:
    Mixing2(const ProcessorGroup* myworld,
            const MaterialManagerP& materialManager,
            const ProblemSpecP& params);
    
    virtual ~Mixing2();
    

    virtual void problemSetup(GridP& grid,
                               const bool isRestart);
    
    virtual void scheduleInitialize(SchedulerP&,
                                    const LevelP& level);

    virtual void restartInitialize() {}
      
    virtual void scheduleComputeStableTimeStep(SchedulerP&,
                                               const LevelP& level);
                                  
    virtual void scheduleComputeModelSources(SchedulerP&,
                                                   const LevelP& level);
                                             
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
                             DataWarehouse* new_dw);

    Mixing2(const Mixing2&);
    Mixing2& operator=(const Mixing2&);

    ProblemSpecP d_params;

    ICELabel* Ilb;
    
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
      std::string name;
      VarLabel* massFraction_CCLabel;
      VarLabel* massFraction_source_CCLabel;
      std::vector<Region*> regions;
    };

    std::vector<Stream*> streams;
    std::map<std::string, Stream*> names;

    Cantera::IdealGasMix* gas;
    Cantera::Reactor* reactor;
  };
}

#endif

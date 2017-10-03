/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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


#ifndef Packages_Uintah_CCA_Components_Examples_Mixing3_h
#define Packages_Uintah_CCA_Components_Examples_Mixing3_h

#include <CCA/Ports/ModelInterface.h>
#include <Core/Grid/Variables/ComputeSet.h>
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
    
    virtual void problemSetup(GridP& grid, SimulationStateP& sharedState,
                              ModelSetup* setup, const bool isRestart);
    
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
    SimulationStateP sharedState;

    double dtemp;
    double dpress;
    double dmf;
    double dtfactor;
    Uintah::FastHashTable<M3Key> table;
    double lookup(int nsp, int idt, int itemp, int ipress, int* imf, double* outmf);
    long nlook;
    long nmiss;
  };
}

#endif

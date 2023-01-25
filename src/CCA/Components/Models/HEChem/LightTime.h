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


#ifndef Packages_Uintah_CCA_Components_Models_LightTime_h
#define Packages_Uintah_CCA_Components_Models_LightTime_h

#include <CCA/Components/Models/HEChem/HEChemModel.h>

#include <Core/Grid/Variables/ComputeSet.h>

namespace Uintah {
  
  class ICELabel;
  
/**************************************

CLASS
   LightTime
  

GENERAL INFORMATION

   LightTime.h

   Jim Guilkey
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   
KEYWORDS
   "Program Burn" "Lighting Time Model"

DESCRIPTION

This model was constructed based on a collection of notes provided
to Jim Guilkey by Peter Vitello at LLNL.  These are excerpts from national
lab code manuals such as HEMP and Kovec, but don't contain enough information
to pull actual references.

The model is pretty straightforward, on specifies the reactant material
and the product material, indicates an origin of the detonation and a detonation
velocity.  Cells start to burn once the distance from the origin to the cell
center, divided by the detonation velocity, meets or exceeds the elapsed
simulation time.  One can also specify an origin as a plane rather than a point.

WARNING

****************************************/

  class LightTime : public HEChemModel {
  public:
    LightTime(const ProcessorGroup* myworld,
              const MaterialManagerP& materialManager,
              const ProblemSpecP& params);
    
    virtual ~LightTime();

    virtual void outputProblemSpec(ProblemSpecP& ps);

    virtual void problemSetup(GridP& grid,
                               const bool isRestart);
      
    virtual void scheduleInitialize(SchedulerP&,
                                    const LevelP& level);

    void initialize(const ProcessorGroup*,
                    const PatchSubset* patches,
                    const MaterialSubset* matls,
                    DataWarehouse*,
                    DataWarehouse*);

    virtual void restartInitialize() {}
      
    virtual void scheduleComputeStableTimeStep(SchedulerP&,
                                               const LevelP& level);
      
    virtual void scheduleComputeModelSources(SchedulerP&,
                                                   const LevelP& level);
                                             
    virtual void scheduleErrorEstimate(const LevelP& coarseLevel,
                                       SchedulerP& sched);

  private:    
    void computeModelSources(const ProcessorGroup*, 
                             const PatchSubset* patches,
                             const MaterialSubset* matls, 
                             DataWarehouse*, 
                             DataWarehouse* new_dw);
                             
    void errorEstimate(const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset*,
                          DataWarehouse*,
                          DataWarehouse* new_dw);

    LightTime(const LightTime&);
    LightTime& operator=(const LightTime&);

    const VarLabel* reactedFractionLabel;   // diagnostic labels
    const VarLabel* mag_grad_Fr_Label;   
    const VarLabel* delFLabel;

    ProblemSpecP d_params;
    const Material* matl0;
    const Material* matl1;

    ICELabel* Ilb;
    MaterialSet* mymatls;
    
    double d_D;   // detonation wave velocity
    double d_E0;
    Point  d_start_place;
    Vector d_direction;
    bool   d_react_mixed_cells;
    double d_refineCriteria;
  };
}

#endif

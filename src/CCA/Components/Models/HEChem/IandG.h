/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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


#ifndef Packages_Uintah_CCA_Components_Models_IandG_h
#define Packages_Uintah_CCA_Components_Models_IandG_h

#include <CCA/Components/Models/HEChem/HEChemModel.h>

#include <Core/Grid/Variables/ComputeSet.h>

namespace Uintah {
  class ICELabel;
/**************************************

CLASS
   IandG
  

GENERAL INFORMATION

   IandG.h

   Jim Guilkey
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   
KEYWORDS
   Ignition and Growth

DESCRIPTION
   Model for detontation of HE based on "Sideways Plate Push Test for
   Detonating Explosives", C.M Tarver, W.C. Tao, Chet. G. Lee, Propellant,
   Explosives, Pyrotechnics, 21, 238-246, 1996.
  
WARNING

****************************************/

  class IandG : public HEChemModel {
  public:
    IandG(const ProcessorGroup* myworld,
	  const SimulationStateP& sharedState,
	  const ProblemSpecP& params);
    
    virtual ~IandG();

    virtual void outputProblemSpec(ProblemSpecP& ps);

    virtual void problemSetup(GridP& grid,
                               const bool isRestart);
      
    virtual void scheduleInitialize(SchedulerP&,
                                    const LevelP& level);

    virtual void restartInitialize() {}
      
 
    virtual void scheduleComputeStableTimeStep(SchedulerP&,
                                               const LevelP& level);
      
    virtual void scheduleComputeModelSources(SchedulerP&,
					     const LevelP& level);
                                             
  private:    
    void computeModelSources(const ProcessorGroup*, 
                             const PatchSubset* patches,
			     const MaterialSubset* matls, 
                             DataWarehouse*, 
			     DataWarehouse* new_dw);

    IandG(const IandG&);
    IandG& operator=(const IandG&);

    const VarLabel* reactedFractionLabel;   // diagnostic labels
    const VarLabel* IandGterm1Label;   // diagnostic labels
    const VarLabel* IandGterm2Label;   // diagnostic labels
    const VarLabel* IandGterm3Label;   // diagnostic labels

    ProblemSpecP d_params;
    const Material* matl0;
    const Material* matl1;

    ICELabel* Ilb;
    MaterialSet* mymatls;
    
    double d_I;
    double d_G1;
    double d_G2;
    double d_a;
    double d_b;
    double d_c;
    double d_d;
    double d_e;
    double d_g;
    double d_x;
    double d_y;
    double d_z;
    double d_Figmax;
    double d_FG1max;
    double d_FG2min;
    double d_rho0;
    double d_E0;
    double d_threshold_pressure;
  };
}

#endif
